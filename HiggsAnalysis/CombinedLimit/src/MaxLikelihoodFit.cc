#include "../interface/MaxLikelihoodFit.h"
#include "RooRealVar.h"
#include "RooArgSet.h"
#include "RooRandom.h"
#include "RooDataSet.h"
#include "RooFitResult.h"
#include "RooPlot.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TH2.h"
#include "TFile.h"
#include <RooStats/ModelConfig.h>
#include "../interface/Combine.h"
#include "../interface/ProfileLikelihood.h"
#include "../interface/CloseCoutSentry.h"
#include "../interface/utils.h"


#include <Math/MinimizerOptions.h>

using namespace RooStats;

std::string MaxLikelihoodFit::name_       = "";
std::string MaxLikelihoodFit::minimizerAlgo_ = "Minuit2,minimize";
float       MaxLikelihoodFit::minimizerTolerance_ = 1e-4;
int         MaxLikelihoodFit::minimizerStrategy_  = 1;
float       MaxLikelihoodFit::preFitValue_ = 1.0;
std::string MaxLikelihoodFit::minos_       = "poi";
std::string MaxLikelihoodFit::out_;
bool        MaxLikelihoodFit::makePlots_ = false;
std::string MaxLikelihoodFit::signalPdfNames_     = "*signal*";
std::string MaxLikelihoodFit::backgroundPdfNames_ = "*background*";


MaxLikelihoodFit::MaxLikelihoodFit() :
    LimitAlgo("Profile Likelihood specific options")
{
    options_.add_options()
        ("minimizerAlgo",      boost::program_options::value<std::string>(&minimizerAlgo_)->default_value(minimizerAlgo_), "Choice of minimizer (Minuit vs Minuit2)")
        ("minimizerTolerance", boost::program_options::value<float>(&minimizerTolerance_)->default_value(minimizerTolerance_),  "Tolerance for minimizer")
        ("minimizerStragegy",  boost::program_options::value<int>(&minimizerStrategy_)->default_value(minimizerStrategy_),      "Stragegy for minimizer")
        ("preFitValue",        boost::program_options::value<float>(&preFitValue_)->default_value(preFitValue_),  "Value of signal strength for pre-fit plots")
        ("minos",              boost::program_options::value<std::string>(&minos_)->default_value(minos_), "Compute MINOS errors for: 'none', 'poi', 'all'")
        ("out",                boost::program_options::value<std::string>(&out_)->default_value(out_), "Directory to put output in")
        ("plots",              "Make plots")
        ("signalPdfNames",     boost::program_options::value<std::string>(&signalPdfNames_)->default_value(signalPdfNames_), "Names of signal pdfs in plots (separated by ,)")
        ("backgroundPdfNames", boost::program_options::value<std::string>(&backgroundPdfNames_)->default_value(backgroundPdfNames_), "Names of background pdfs in plots (separated by ',')")
    ;
}

void MaxLikelihoodFit::applyOptions(const boost::program_options::variables_map &vm) 
{
    makePlots_ = vm.count("plots");
    name_ = vm["name"].defaulted() ?  std::string() : vm["name"].as<std::string>();
}

bool MaxLikelihoodFit::run(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double &limit, double &limitErr, const double *hint) { 
  ProfileLikelihood::MinimizerSentry minimizerConfig(minimizerAlgo_, minimizerTolerance_);
  CloseCoutSentry sentry(verbose < 0);

  std::auto_ptr<TFile> fitOut(TFile::Open((out_+"/mlfit"+name_+".root").c_str(), "RECREATE"));

  RooRealVar *r = dynamic_cast<RooRealVar *>(mc_s->GetParametersOfInterest()->first());

  utils::tdrStyle();
  TCanvas *c1 = new TCanvas("c1","c1");

  // Make pre-plots before the fit
  r->setVal(preFitValue_);
  if (makePlots_) {
      std::vector<RooPlot *> plots = utils::makePlots(*mc_s->GetPdf(), data, signalPdfNames_.c_str(), backgroundPdfNames_.c_str());
      for (std::vector<RooPlot *>::iterator it = plots.begin(), ed = plots.end(); it != ed; ++it) {
          (*it)->Draw(); 
          c1->Print((out_+"/"+(*it)->GetName()+"_prefit.png").c_str());
          fitOut->WriteTObject(*it, (std::string((*it)->GetName())+"_prefit").c_str());
      }
  }

  // Determine pre-fit values of nuisance parameters
  const RooArgSet *nuis      = mc_s->GetNuisanceParameters();
  const RooArgSet *globalObs = mc_s->GetGlobalObservables();
  std::auto_ptr<RooAbsPdf> nuisancePdf(utils::makeNuisancePdf(*mc_s));
  if (nuis && globalObs) {
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
      fitOut->WriteTObject(res_prefit, "nuisances_prefit_res");
      fitOut->WriteTObject(nuis->snapshot(), "nuisances_prefit");
  } else if (nuis) {
      fitOut->WriteTObject(nuis->snapshot(), "nuisances_prefit");
  }

  
  RooFitResult *res_b = 0, *res_s = 0;
  const RooCmdArg &constCmdArg_s = withSystematics  ? RooFit::Constrain(*mc_s->GetNuisanceParameters()) : RooFit::NumCPU(1); // use something dummy 
  const RooCmdArg &minosCmdArg = minos_ == "poi" ?  RooFit::Minos(*mc_s->GetParametersOfInterest())   : RooFit::Minos(minos_ != "none"); 
  r->setVal(0.0); r->setConstant(true);
  {
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
      if (verbose > 0) res_b->Print("V");
      fitOut->WriteTObject(res_b, "fit_b");

      if (makePlots_) {
          std::vector<RooPlot *> plots = utils::makePlots(*mc_b->GetPdf(), data, 0, backgroundPdfNames_.c_str());
          for (std::vector<RooPlot *>::iterator it = plots.begin(), ed = plots.end(); it != ed; ++it) {
              c1->cd(); (*it)->Draw(); 
              c1->Print((out_+"/"+(*it)->GetName()+"_fit_b.png").c_str());
              fitOut->WriteTObject(*it, (std::string((*it)->GetName())+"_fit_b").c_str());
          }
      }

      TH2 *corr = res_b->correlationHist();
      c1->SetLeftMargin(0.25);  c1->SetBottomMargin(0.25);
      corr->SetTitle("Correlation matrix of fit parameters");
      gStyle->SetPaintTextFormat(res_b->floatParsFinal().getSize() > 10 ? ".1f" : ".2f");
      gStyle->SetOptStat(0);
      corr->SetMarkerSize(res_b->floatParsFinal().getSize() > 10 ? 2 : 1);
      corr->Draw("COLZ TEXT");
      if (makePlots_) c1->Print((out_+"/covariance_fit_b.png").c_str());
      c1->SetLeftMargin(0.16);  c1->SetBottomMargin(0.13);
      fitOut->WriteTObject(corr, "covariance_fit_b");
  }

  r->setVal(preFitValue_); r->setConstant(false);
  {
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
      if (verbose > 0) res_s->Print("V");
      fitOut->WriteTObject(res_s, "fit_s");

      if (makePlots_) {
          std::vector<RooPlot *> plots = utils::makePlots(*mc_s->GetPdf(), data, 0, backgroundPdfNames_.c_str());
          for (std::vector<RooPlot *>::iterator it = plots.begin(), ed = plots.end(); it != ed; ++it) {
              c1->cd(); (*it)->Draw(); 
              c1->Print((out_+"/"+(*it)->GetName()+"_fit_s.png").c_str());
              fitOut->WriteTObject(*it, (std::string((*it)->GetName())+"_fit_s").c_str());
          }
      }


      TH2 *corr = res_s->correlationHist();
      c1->SetLeftMargin(0.25);  c1->SetBottomMargin(0.25);
      corr->SetTitle("Correlation matrix of fit parameters");
      gStyle->SetPaintTextFormat(res_s->floatParsFinal().getSize() > 10 ? ".1f" : ".2f");
      gStyle->SetOptStat(0);
      corr->SetMarkerSize(res_s->floatParsFinal().getSize() > 10 ? 2 : 1);
      corr->Draw("COLZ TEXT");
      if (makePlots_) c1->Print((out_+"/covariance_fit_s.png").c_str());
      c1->SetLeftMargin(0.16);  c1->SetBottomMargin(0.13);
      fitOut->WriteTObject(corr, "covariance_fit_s");
  }

  fitOut.release()->Close();
  return res_s != 0;
}


