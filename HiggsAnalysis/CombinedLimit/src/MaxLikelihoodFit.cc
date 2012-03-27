#include "../interface/MaxLikelihoodFit.h"
#include "RooRealVar.h"
#include "RooArgSet.h"
#include "RooRandom.h"
#include "RooDataSet.h"
#include "RooFitResult.h"
#include "RooSimultaneous.h"
#include "RooAddPdf.h"
#include "RooProdPdf.h"
#include "RooConstVar.h"
#include "RooPlot.h"
#include "../interface/RooMinimizerOpt.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TH2.h"
#include "TFile.h"
#include <RooStats/ModelConfig.h>
#include "../interface/Combine.h"
#include "../interface/ProfileLikelihood.h"
#include "../interface/ProfiledLikelihoodRatioTestStatExt.h"
#include "../interface/CloseCoutSentry.h"
#include "../interface/utils.h"


#include <Math/MinimizerOptions.h>

using namespace RooStats;

std::string MaxLikelihoodFit::name_ = "";
std::string MaxLikelihoodFit::minos_ = "poi";
std::string MaxLikelihoodFit::out_ = ".";
bool        MaxLikelihoodFit::makePlots_ = false;
float       MaxLikelihoodFit::rebinFactor_ = 1.0;
std::string MaxLikelihoodFit::signalPdfNames_     = "shapeSig*";
std::string MaxLikelihoodFit::backgroundPdfNames_ = "shapeBkg*";
bool        MaxLikelihoodFit::saveNormalizations_ = false;
bool        MaxLikelihoodFit::justFit_ = false;


MaxLikelihoodFit::MaxLikelihoodFit() :
    FitterAlgoBase("MaxLikelihoodFit specific options")
{
    options_.add_options()
        ("minos",              boost::program_options::value<std::string>(&minos_)->default_value(minos_), "Compute MINOS errors for: 'none', 'poi', 'all'")
        ("out",                boost::program_options::value<std::string>(&out_)->default_value(out_), "Directory to put output in")
        ("plots",              "Make plots")
        ("rebinFactor",        boost::program_options::value<float>(&rebinFactor_)->default_value(rebinFactor_), "Rebin by this factor before plotting (does not affect fitting!)")
        ("signalPdfNames",     boost::program_options::value<std::string>(&signalPdfNames_)->default_value(signalPdfNames_), "Names of signal pdfs in plots (separated by ,)")
        ("backgroundPdfNames", boost::program_options::value<std::string>(&backgroundPdfNames_)->default_value(backgroundPdfNames_), "Names of background pdfs in plots (separated by ',')")
        ("saveNormalizations",  "Save post-fit normalizations of all components of the pdfs")
        ("justFit",  "Just do the S+B fit, don't do the B-only one, don't save output file")
   ;
}

void MaxLikelihoodFit::applyOptions(const boost::program_options::variables_map &vm) 
{
    applyOptionsBase(vm);
    makePlots_ = vm.count("plots");
    name_ = vm["name"].defaulted() ?  std::string() : vm["name"].as<std::string>();
    saveNormalizations_  = vm.count("saveNormalizations");
    justFit_  = vm.count("justFit");
    if (justFit_) { out_ = "none"; makePlots_ = false; saveNormalizations_ = false; }
}

bool MaxLikelihoodFit::runSpecific(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double &limit, double &limitErr, const double *hint) { 
  std::auto_ptr<TFile> fitOut;
  if (!justFit_ && out_ != "none") fitOut.reset(TFile::Open((out_+"/mlfit"+name_+".root").c_str(), "RECREATE"));

  RooRealVar *r = dynamic_cast<RooRealVar *>(mc_s->GetParametersOfInterest()->first());

  TCanvas *c1 = 0;
  if (makePlots_) {
      utils::tdrStyle();
      c1 = new TCanvas("c1","c1");
  }

  // Make pre-plots before the fit
  r->setVal(preFitValue_);
  if (makePlots_) {
      std::vector<RooPlot *> plots = utils::makePlots(*mc_s->GetPdf(), data, signalPdfNames_.c_str(), backgroundPdfNames_.c_str(), rebinFactor_);
      for (std::vector<RooPlot *>::iterator it = plots.begin(), ed = plots.end(); it != ed; ++it) {
          (*it)->Draw(); 
          c1->Print((out_+"/"+(*it)->GetName()+"_prefit.png").c_str());
          if (fitOut.get()) fitOut->WriteTObject(*it, (std::string((*it)->GetName())+"_prefit").c_str());
      }
  }

  // Determine pre-fit values of nuisance parameters
  const RooArgSet *nuis      = mc_s->GetNuisanceParameters();
  const RooArgSet *globalObs = mc_s->GetGlobalObservables();
  if (!justFit_ && nuis && globalObs) {
      std::auto_ptr<RooAbsPdf> nuisancePdf(utils::makeNuisancePdf(*mc_s));
      std::auto_ptr<RooDataSet> globalData(new RooDataSet("globalData","globalData", *globalObs));
      globalData->add(*globalObs);
      RooFitResult *res_prefit = 0;
      {     
            CloseCoutSentry sentry(verbose < 2);
            res_prefit = nuisancePdf->fitTo(*globalData,
            RooFit::Save(1),
            RooFit::Minimizer(ROOT::Math::MinimizerOptions::DefaultMinimizerType().c_str(), ROOT::Math::MinimizerOptions::DefaultMinimizerAlgo().c_str()),
            RooFit::Strategy(minimizerStrategy_),
            RooFit::Minos(minos_ == "all")
            );
      }
      if (fitOut.get()) fitOut->WriteTObject(res_prefit, "nuisances_prefit_res");
      if (fitOut.get()) fitOut->WriteTObject(nuis->snapshot(), "nuisances_prefit");
  } else if (nuis) {
      if (fitOut.get()) fitOut->WriteTObject(nuis->snapshot(), "nuisances_prefit");
  }

  
  RooFitResult *res_b = 0, *res_s = 0;
  const RooCmdArg &constCmdArg_s = withSystematics  ? RooFit::Constrain(*mc_s->GetNuisanceParameters()) : RooFit::NumCPU(1); // use something dummy 
  const RooCmdArg &minosCmdArg = minos_ == "poi" ?  RooFit::Minos(*mc_s->GetParametersOfInterest())   : RooFit::Minos(minos_ != "none"); 
  w->loadSnapshot("clean");
  r->setVal(0.0); r->setConstant(true);

  if (justFit_) { 
    // skip b-only fit
  } else if (minos_ != "all") {
    RooArgList minos; 
    res_b = doFit(*mc_s->GetPdf(), data, minos, constCmdArg_s, /*hesse=*/true); 
  } else {
    CloseCoutSentry sentry(verbose < 2);
    res_b = mc_s->GetPdf()->fitTo(data, 
            RooFit::Save(1), 
            RooFit::Minimizer(ROOT::Math::MinimizerOptions::DefaultMinimizerType().c_str(), ROOT::Math::MinimizerOptions::DefaultMinimizerAlgo().c_str()), 
            RooFit::Strategy(minimizerStrategy_),
            RooFit::Extended(mc_s->GetPdf()->canBeExtended()), 
            constCmdArg_s, minosCmdArg
            );
  }

  if (res_b) { 
      if (verbose > 1) res_b->Print("V");
      if (fitOut.get()) fitOut->WriteTObject(res_b, "fit_b");

      if (makePlots_) {
          std::vector<RooPlot *> plots = utils::makePlots(*mc_b->GetPdf(), data, signalPdfNames_.c_str(), backgroundPdfNames_.c_str(), rebinFactor_);
          for (std::vector<RooPlot *>::iterator it = plots.begin(), ed = plots.end(); it != ed; ++it) {
              c1->cd(); (*it)->Draw(); 
              c1->Print((out_+"/"+(*it)->GetName()+"_fit_b.png").c_str());
              if (fitOut.get()) fitOut->WriteTObject(*it, (std::string((*it)->GetName())+"_fit_b").c_str());
          }
      }

      if (saveNormalizations_) {
          RooArgSet *norms = new RooArgSet();
          norms->setName("norm_fit_b");
          getNormalizations(mc_s->GetPdf(), *mc_s->GetObservables(), *norms);
          if (fitOut.get()) fitOut->WriteTObject(norms, "norm_fit_b");
      }

      TH2 *corr = res_b->correlationHist();
      if (makePlots_)  {
          c1->SetLeftMargin(0.25);  c1->SetBottomMargin(0.25);
          corr->SetTitle("Correlation matrix of fit parameters");
          gStyle->SetPaintTextFormat(res_b->floatParsFinal().getSize() > 10 ? ".1f" : ".2f");
          gStyle->SetOptStat(0);
          corr->SetMarkerSize(res_b->floatParsFinal().getSize() > 10 ? 2 : 1);
          corr->Draw("COLZ TEXT");
          c1->Print((out_+"/covariance_fit_b.png").c_str());
          c1->SetLeftMargin(0.16);  c1->SetBottomMargin(0.13);
      }
      if (fitOut.get()) fitOut->WriteTObject(corr, "covariance_fit_b");
  }

  w->loadSnapshot("clean");
  r->setVal(preFitValue_); r->setConstant(false);
  if (minos_ != "all") {
    RooArgList minos; if (minos_ == "poi") minos.add(*r);
    res_s = doFit(*mc_s->GetPdf(), data, minos, constCmdArg_s, /*hesse=*/true); 
  } else {
    CloseCoutSentry sentry(verbose < 2);
    res_s = mc_s->GetPdf()->fitTo(data, 
            RooFit::Save(1), 
            RooFit::Minimizer(ROOT::Math::MinimizerOptions::DefaultMinimizerType().c_str(), ROOT::Math::MinimizerOptions::DefaultMinimizerAlgo().c_str()), 
            RooFit::Strategy(minimizerStrategy_),
            RooFit::Extended(mc_s->GetPdf()->canBeExtended()), 
            constCmdArg_s, minosCmdArg
            );
  }
  if (res_s) { 
      limit    = r->getVal();
      limitErr = r->getError();
      if (verbose > 1) res_s->Print("V");
      if (fitOut.get()) fitOut->WriteTObject(res_s, "fit_s");

      if (makePlots_) {
          std::vector<RooPlot *> plots = utils::makePlots(*mc_s->GetPdf(), data, signalPdfNames_.c_str(), backgroundPdfNames_.c_str(), rebinFactor_);
          for (std::vector<RooPlot *>::iterator it = plots.begin(), ed = plots.end(); it != ed; ++it) {
              c1->cd(); (*it)->Draw(); 
              c1->Print((out_+"/"+(*it)->GetName()+"_fit_s.png").c_str());
              if (fitOut.get()) fitOut->WriteTObject(*it, (std::string((*it)->GetName())+"_fit_s").c_str());
          }
      }

      if (saveNormalizations_) {
          RooArgSet *norms = new RooArgSet();
          norms->setName("norm_fit_s");
          getNormalizations(mc_s->GetPdf(), *mc_s->GetObservables(), *norms);
          if (fitOut.get()) fitOut->WriteTObject(norms, "norm_fit_s");
      }

      TH2 *corr = res_s->correlationHist();
      if (makePlots_)  {
          c1->SetLeftMargin(0.25);  c1->SetBottomMargin(0.25);
          corr->SetTitle("Correlation matrix of fit parameters");
          gStyle->SetPaintTextFormat(res_s->floatParsFinal().getSize() > 10 ? ".1f" : ".2f");
          gStyle->SetOptStat(0);
          corr->SetMarkerSize(res_s->floatParsFinal().getSize() > 10 ? 2 : 1);
          corr->Draw("COLZ TEXT");
          c1->Print((out_+"/covariance_fit_s.png").c_str());
          c1->SetLeftMargin(0.16);  c1->SetBottomMargin(0.13);
      }
      if (fitOut.get()) fitOut->WriteTObject(corr, "covariance_fit_s");
  }

  if (res_s) {
      RooRealVar *rf = dynamic_cast<RooRealVar*>(res_s->floatParsFinal().find(r->GetName()));
      double bestFitVal = rf->getVal();

      double hiErr = +(rf->hasRange("err68") ? rf->getMax("err68") - bestFitVal : rf->getAsymErrorHi());
      double loErr = -(rf->hasRange("err68") ? rf->getMin("err68") - bestFitVal : rf->getAsymErrorLo());
      double maxError = std::max<double>(std::max<double>(hiErr, loErr), rf->getError());

      if (fabs(hiErr) < 0.001*maxError) hiErr = -bestFitVal + rf->getMax();
      if (fabs(loErr) < 0.001*maxError) loErr = +bestFitVal - rf->getMin();

      double hiErr95 = +(do95_ && rf->hasRange("err95") ? rf->getMax("err95") - bestFitVal : 0);
      double loErr95 = -(do95_ && rf->hasRange("err95") ? rf->getMin("err95") - bestFitVal : 0);

      limit = bestFitVal;  limitErr = 0;
      Combine::commitPoint(/*expected=*/true, /*quantile=*/0.5);
      limit = bestFitVal - loErr; limitErr = 0;
      Combine::commitPoint(/*expected=*/true, /*quantile=*/0.16);
      limit = bestFitVal + hiErr; limitErr = 0;
      Combine::commitPoint(/*expected=*/true, /*quantile=*/0.84);
      if (do95_ && rf->hasRange("err95")) {
        limit = rf->getMax("err95"); Combine::commitPoint(/*expected=*/true, /*quantile=*/0.975);
        limit = rf->getMin("err95"); Combine::commitPoint(/*expected=*/true, /*quantile=*/0.025);
      }

      limit = bestFitVal;
      limitErr = maxError;
      std::cout << "\n --- MaxLikelihoodFit ---" << std::endl;
      std::cout << "Best fit " << r->GetName() << ": " << rf->getVal() << "  "<<  -loErr << "/+" << +hiErr << "  (68% CL)" << std::endl;
      if (do95_) {
        std::cout << "         " << r->GetName() << ": " << rf->getVal() << "  "<<  -loErr95 << "/+" << +hiErr95 << "  (95% CL)" << std::endl;
      }
  } else {
      std::cout << "\n --- MaxLikelihoodFit ---" << std::endl;
      std::cout << "Fit failed."  << std::endl;
  }
  if (fitOut.get()) fitOut.release()->Close();
  return res_s != 0;
}

void MaxLikelihoodFit::getNormalizations(RooAbsPdf *pdf, const RooArgSet &obs, RooArgSet &out) {
    RooSimultaneous *sim = dynamic_cast<RooSimultaneous *>(pdf);
    if (sim != 0) {
        RooAbsCategoryLValue &cat = const_cast<RooAbsCategoryLValue &>(sim->indexCat());
        for (int i = 0, n = cat.numBins((const char *)0); i < n; ++i) {
            cat.setBin(i);
            RooAbsPdf *pdfi = sim->getPdf(cat.getLabel());
            if (pdfi) getNormalizations(pdfi, obs, out);
        }        
        return;
    }
    RooProdPdf *prod = dynamic_cast<RooProdPdf *>(pdf);
    if (prod != 0) {
        RooArgList list(prod->pdfList());
        for (int i = 0, n = list.getSize(); i < n; ++i) {
            RooAbsPdf *pdfi = (RooAbsPdf *) list.at(i);
            if (pdfi->dependsOn(obs)) getNormalizations(pdfi, obs, out);
        }
        return;
    }
    RooAddPdf *add = dynamic_cast<RooAddPdf *>(pdf);
    if (add != 0) {
        RooArgList list(add->coefList());
        for (int i = 0, n = list.getSize(); i < n; ++i) {
            RooAbsReal *coeff = (RooAbsReal *) list.at(i);
            out.addOwned(*(new RooConstVar(coeff->GetName(), "", coeff->getVal())));
        }
        return;
    }
}


