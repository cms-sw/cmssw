#include "../interface/GoodnessOfFit.h"
#include <RooRealVar.h>
#include <RooArgSet.h>
#include <RooRandom.h>
#include <RooDataSet.h>
#include <RooFitResult.h>
#include <RooPlot.h>
#include <RooProdPdf.h>
#include <RooSimultaneous.h>
#include <RooAddPdf.h>
#include <RooConstVar.h>
#include <RooDataSet.h>
#include <RooDataHist.h>
#include <RooHistPdf.h>
#include <TCanvas.h>
#include <TStyle.h>
#include <TH2.h>
#include <TFile.h>
#include <RooStats/ModelConfig.h>
#include "../interface/Combine.h"
#include "../interface/ProfileLikelihood.h"
#include "../interface/CloseCoutSentry.h"
#include "../interface/RooSimultaneousOpt.h"
#include "../interface/utils.h"


#include <Math/MinimizerOptions.h>

using namespace RooStats;

std::string GoodnessOfFit::algo_;
std::string GoodnessOfFit::minimizerAlgo_ = "Minuit2";
float       GoodnessOfFit::minimizerTolerance_ = 1e-4;
int         GoodnessOfFit::minimizerStrategy_  = 1;
float       GoodnessOfFit::mu_ = 0.0;
bool        GoodnessOfFit::fixedMu_ = false;

GoodnessOfFit::GoodnessOfFit() :
    LimitAlgo("GoodnessOfFit specific options")
{
    options_.add_options()
        ("algorithm",          boost::program_options::value<std::string>(&algo_), "Goodness of fit algorithm. Currently, the only option is 'saturated' (which works for binned models only)")
        ("minimizerAlgo",      boost::program_options::value<std::string>(&minimizerAlgo_)->default_value(minimizerAlgo_), "Choice of minimizer (Minuit vs Minuit2)")
        ("minimizerTolerance", boost::program_options::value<float>(&minimizerTolerance_)->default_value(minimizerTolerance_),  "Tolerance for minimizer")
        ("minimizerStrategy",  boost::program_options::value<int>(&minimizerStrategy_)->default_value(minimizerStrategy_),      "Stragegy for minimizer")
        ("fixedSignalStrength", boost::program_options::value<float>(&mu_)->default_value(mu_),  "Compute the goodness of fit for a fixed signal strength. If not specified, it's left floating")
    ;
}

void GoodnessOfFit::applyOptions(const boost::program_options::variables_map &vm) 
{
    fixedMu_ = !vm["fixedSignalStrength"].defaulted();
    if (algo_ == "saturated") std::cout << "Will use saturated models to compute goodness of fit for a binned likelihood" << std::endl;
    else throw std::invalid_argument("GoodnessOfFit: algorithm "+algo_+" not supported");
}

bool GoodnessOfFit::run(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double &limit, double &limitErr, const double *hint) { 
  ProfileLikelihood::MinimizerSentry minimizerConfig(minimizerAlgo_, minimizerTolerance_);

  RooRealVar *r = dynamic_cast<RooRealVar *>(mc_s->GetParametersOfInterest()->first());
  if (fixedMu_) { r->setVal(mu_); r->setConstant(true); }
  if (algo_ == "saturated") return runSaturatedModel(w, mc_s, mc_b, data, limit, limitErr, hint);
  return false;  
}

bool GoodnessOfFit::runSaturatedModel(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double &limit, double &limitErr, const double *hint) { 
  RooAbsPdf *pdf_nominal = mc_s->GetPdf();
  // now I need to make the saturated pdf
  std::auto_ptr<RooAbsPdf> saturated;
  // factorize away constraints anyway
  RooArgList constraints;
  RooAbsPdf *obsOnlyPdf = utils::factorizePdf(*mc_s->GetObservables(), *pdf_nominal, constraints);
  // case 1:
  RooSimultaneous *sim = dynamic_cast<RooSimultaneous *>(obsOnlyPdf);
  if (sim) {
      RooAbsCategoryLValue *cat = (RooAbsCategoryLValue *) sim->indexCat().Clone();
      std::auto_ptr<TList> datasets(data.split(*cat, true));
      int nbins = cat->numBins((const char *)0);
      RooArgSet newPdfs;
      TString satname = TString::Format("%s_saturated", sim->GetName());
      RooSimultaneous *satsim = (typeid(*sim) == typeid(RooSimultaneousOpt)) ? new RooSimultaneousOpt(satname, "", *cat) : new RooSimultaneous(satname, "", *cat); 
      for (int ic = 0, nc = nbins; ic < nc; ++ic) {
          cat->setBin(ic);
          RooAbsPdf *pdfi = sim->getPdf(cat->getLabel());
          if (pdfi == 0) continue;
          RooAbsData *datai = (RooAbsData *) datasets->FindObject(cat->getLabel());
          if (datai == 0) throw std::runtime_error(std::string("Error: missing dataset for category label ")+cat->getLabel());
          RooAbsPdf *saturatedPdfi = makeSaturatedPdf(*datai);
          //delete datai;
          if (constraints.getSize() > 0) {
            RooArgList terms(constraints); terms.add(*saturatedPdfi);
            RooProdPdf *prodpdf = new RooProdPdf(TString::Format("%s_constr", saturatedPdfi->GetName()), "", terms);
            prodpdf->addOwnedComponents(RooArgSet(*saturatedPdfi));
            saturatedPdfi = prodpdf;
          }
          satsim->addPdf(*saturatedPdfi, cat->getLabel());
          satsim->addOwnedComponents(RooArgSet(*saturatedPdfi));
      }
      saturated.reset(satsim);
  } else {
      RooAbsPdf *saturatedPdfi = makeSaturatedPdf(data);
      if (constraints.getSize() > 0) {
          RooArgList terms(constraints); terms.add(*saturatedPdfi);
          RooProdPdf *prodpdf = new RooProdPdf(TString::Format("%s_constr", saturatedPdfi->GetName()), "", terms);
          prodpdf->addOwnedComponents(RooArgSet(*saturatedPdfi));
          saturatedPdfi = prodpdf;
      }
      saturated.reset(saturatedPdfi);
  }

  CloseCoutSentry sentry(verbose < 2);
  // let's assume fits converge, for a while
  const RooCmdArg &minim = RooFit::Minimizer(ROOT::Math::MinimizerOptions::DefaultMinimizerType().c_str(),
                                             ROOT::Math::MinimizerOptions::DefaultMinimizerAlgo().c_str());
  std::auto_ptr<RooFitResult> result_nominal(pdf_nominal->fitTo(data, RooFit::Save(1), minim, RooFit::Strategy(minimizerStrategy_), RooFit::Hesse(0), RooFit::Constrain(*mc_s->GetNuisanceParameters())));
  std::auto_ptr<RooFitResult> result_saturated(saturated->fitTo(data, RooFit::Save(1), minim, RooFit::Strategy(minimizerStrategy_), RooFit::Hesse(0), RooFit::Constrain(*mc_s->GetNuisanceParameters())));
  sentry.clear();

  saturated.reset();
  for (int i = 0, n = tempData_.size(); i < n; ++i) delete tempData_[i]; 
  tempData_.clear();

  if (result_nominal.get()   == 0) return false;
  if (result_saturated.get() == 0) return false;

  double nll_nominal   = result_nominal->minNll();
  double nll_saturated = result_saturated->minNll();
  if (fabs(nll_nominal) > 1e10 || fabs(nll_saturated) > 1e10) return false;
  limit = 2*(nll_nominal-nll_saturated);

  std::cout << "\n --- GoodnessOfFit --- " << std::endl;
  std::cout << "Best fit test statistic: " << limit << std::endl;
  return true;
}

RooAbsPdf * GoodnessOfFit::makeSaturatedPdf(RooAbsData &data) {
  if (verbose > 1) std::cout << "Generating saturated model for " << data.GetName() << std::endl;
  RooDataHist *rdh = new RooDataHist(TString::Format("%s_binned", data.GetName()), "", *data.get(), data); tempData_.push_back(rdh);
  if (verbose > 1) utils::printRDH(rdh);
  RooHistPdf *hpdf = new RooHistPdf(TString::Format("%s_shape", data.GetName()), "", *rdh->get(), *rdh);
  RooConstVar *norm = new RooConstVar(TString::Format("%s_norm", data.GetName()), "", data.sumEntries());
  // we use RooAddPdf because this works with CachingNLL
  RooAddPdf *ret = new RooAddPdf(TString::Format("%s_saturated", data.GetName()), "", RooArgList(*hpdf), RooArgList(*norm));
  ret->addOwnedComponents(RooArgSet(*norm));
  ret->addOwnedComponents(RooArgSet(*hpdf));
  return ret;
}


