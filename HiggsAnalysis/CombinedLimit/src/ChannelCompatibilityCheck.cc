#include "../interface/ChannelCompatibilityCheck.h"
#include <TFile.h>
#include <RooRealVar.h>
#include <RooArgSet.h>
#include <RooRandom.h>
#include <RooDataSet.h>
#include <RooFitResult.h>
#include <RooCustomizer.h>
#include <RooSimultaneous.h>
#include <RooStats/ModelConfig.h>
#include "../interface/Combine.h"
#include "../interface/ProfileLikelihood.h"
#include "../interface/CloseCoutSentry.h"
#include "../interface/RooSimultaneousOpt.h"
#include "../interface/utils.h"


#include <Math/MinimizerOptions.h>

using namespace RooStats;

std::string ChannelCompatibilityCheck::minimizerAlgo_ = "Minuit2,minimize";
float       ChannelCompatibilityCheck::minimizerTolerance_ = 1e-4;
int         ChannelCompatibilityCheck::minimizerStrategy_  = 1;
float       ChannelCompatibilityCheck::mu_ = 0.0;
bool        ChannelCompatibilityCheck::fixedMu_ = false;
bool        ChannelCompatibilityCheck::saveFitResult_ = false;
std::vector<std::string> ChannelCompatibilityCheck::groups_;

ChannelCompatibilityCheck::ChannelCompatibilityCheck() :
    LimitAlgo("ChannelCompatibilityCheck specific options")
{
    options_.add_options()
        ("minimizerAlgo",      boost::program_options::value<std::string>(&minimizerAlgo_)->default_value(minimizerAlgo_), "Choice of minimizer (Minuit vs Minuit2)")
        ("minimizerTolerance", boost::program_options::value<float>(&minimizerTolerance_)->default_value(minimizerTolerance_),  "Tolerance for minimizer")
        ("minimizerStrategy",  boost::program_options::value<int>(&minimizerStrategy_)->default_value(minimizerStrategy_),      "Stragegy for minimizer")
        ("fixedSignalStrength", boost::program_options::value<float>(&mu_)->default_value(mu_),  "Compute the compatibility for a fixed signal strength. If not specified, it's left floating")
        ("saveFitResult",       "Save fit results in output file")
        ("groups",              boost::program_options::value<std::string>(), "Group channels according to the specified list of expressions (NOT IMPLEMENTED YET)")
    ;
}

void ChannelCompatibilityCheck::applyOptions(const boost::program_options::variables_map &vm) 
{
    fixedMu_ = !vm["fixedSignalStrength"].defaulted();
    saveFitResult_ = vm.count("saveFitResult");
}

bool ChannelCompatibilityCheck::run(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double &limit, double &limitErr, const double *hint) { 
  ProfileLikelihood::MinimizerSentry minimizerConfig(minimizerAlgo_, minimizerTolerance_);

  RooRealVar *r = dynamic_cast<RooRealVar *>(mc_s->GetParametersOfInterest()->first());
  if (fixedMu_) { r->setVal(mu_); r->setConstant(true); }

  RooSimultaneous *sim = dynamic_cast<RooSimultaneous *>(mc_s->GetPdf());
  if (sim == 0) throw std::logic_error("Cannot use ChannelCompatibilityCheck if the pdf is not a RooSimultaneous");

  RooAbsCategoryLValue *cat = (RooAbsCategoryLValue *) sim->indexCat().Clone();
  int nbins = cat->numBins((const char *)0);
  TString satname = TString::Format("%s_freeform", sim->GetName());
  std::auto_ptr<RooSimultaneous> newsim((typeid(*sim) == typeid(RooSimultaneousOpt)) ? new RooSimultaneousOpt(satname, "", *cat) : new RooSimultaneous(satname, "", *cat)); 
  std::vector<std::pair<std::string,std::string> > rs;
  RooArgSet minosVars;
  for (int ic = 0, nc = nbins; ic < nc; ++ic) {
      cat->setBin(ic);
      RooAbsPdf *pdfi = sim->getPdf(cat->getLabel());
      if (pdfi == 0) continue;
      RooCustomizer customizer(*pdfi, "freeform");
      TString riName = TString::Format("_ChannelCompatibilityCheck_%s_%s", r->GetName(), cat->getLabel());
      rs.push_back(std::pair<std::string,std::string>(cat->getLabel(), riName.Data()));
      if (w->var(riName) == 0) {
        w->factory(TString::Format("%s[%g,%g]", riName.Data(), r->getMin(), r->getMax()));
      }
      customizer.replaceArg(*r, *w->var(riName));
      newsim->addPdf((RooAbsPdf&)*customizer.build(), cat->getLabel());
      minosVars.add(*w->var(riName));
  }

  CloseCoutSentry sentry(verbose < 2);
  // let's assume fits converge, for a while
  const RooCmdArg &minim = RooFit::Minimizer(ROOT::Math::MinimizerOptions::DefaultMinimizerType().c_str(),
                                             ROOT::Math::MinimizerOptions::DefaultMinimizerAlgo().c_str());
  std::auto_ptr<RooFitResult> result_nominal(    sim->fitTo(data, RooFit::Save(1), minim, RooFit::Strategy(minimizerStrategy_), RooFit::Minos(RooArgSet(*r)), RooFit::Constrain(*mc_s->GetNuisanceParameters())));
  std::auto_ptr<RooFitResult> result_freeform(newsim->fitTo(data, RooFit::Save(1), minim, RooFit::Strategy(minimizerStrategy_), RooFit::Minos(minosVars),     RooFit::Constrain(*mc_s->GetNuisanceParameters())));
  sentry.clear();

  if (result_nominal.get()  == 0) return false;
  if (result_freeform.get() == 0) return false;

  double nll_nominal   = result_nominal->minNll();
  double nll_freeform = result_freeform->minNll();
  if (fabs(nll_nominal) > 1e10 || fabs(nll_freeform) > 1e10) return false;
  limit = 2*(nll_nominal-nll_freeform);
  
  std::cout << "\n --- ChannelCompatibilityCheck --- " << std::endl;
  if (verbose) {
    if (fixedMu_) { 
        printf("Nominal fit: %s fixed at %7.4f\n", r->GetName(), r->getVal());
    } else {
        RooRealVar *rNominal = (RooRealVar*) result_nominal->floatParsFinal().find(r->GetName());
        printf("Nominal fit  : %s = %7.4f  %+6.4f/%+6.4f\n", r->GetName(), rNominal->getVal(), rNominal->getAsymErrorLo(), rNominal->getAsymErrorHi());
    }
    for (int i = 0, n = rs.size(); i < n; ++i) {
        RooRealVar *ri = (RooRealVar*) result_freeform->floatParsFinal().find(rs[i].second.c_str());
        printf("Alternate fit: %s = %7.4f  %+6.4f/%+6.4f   in channel %s\n", r->GetName(), ri->getVal(), ri->getAsymErrorLo(), ri->getAsymErrorHi(), rs[i].first.c_str());
    }
  }
  std::cout << "Chi2-like compatibility variable: " << limit << std::endl;

  if (saveFitResult_) {
      writeToysHere->GetFile()->WriteTObject(result_nominal.release(),  "fit_nominal"  );
      writeToysHere->GetFile()->WriteTObject(result_freeform.release(), "fit_alternate");
  }
  return true;
}


