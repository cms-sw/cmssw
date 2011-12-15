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

float ChannelCompatibilityCheck::mu_ = 0.0;
bool  ChannelCompatibilityCheck::fixedMu_ = false;
bool  ChannelCompatibilityCheck::saveFitResult_ = true;
bool  ChannelCompatibilityCheck::runMinos_ = true;
std::vector<std::string> ChannelCompatibilityCheck::groups_;

ChannelCompatibilityCheck::ChannelCompatibilityCheck() :
    FitterAlgoBase("ChannelCompatibilityCheck specific options")
{
    options_.add_options()
        ("fixedSignalStrength", boost::program_options::value<float>(&mu_)->default_value(mu_),  "Compute the compatibility for a fixed signal strength. If not specified, it's left floating")
        ("saveFitResult",       "Save fit results in output file")
        ("group,g",             boost::program_options::value<std::vector<std::string> >(&groups_), "Group together channels that contain a given name. Can be used multiple times.")
        ("runMinos", boost::program_options::value<bool>(&runMinos_)->default_value(runMinos_), "Compute also uncertainties using profile likeilhood (MINOS or robust variants of it)")
    ;
}

void ChannelCompatibilityCheck::applyOptions(const boost::program_options::variables_map &vm) 
{
    applyOptionsBase(vm);
    fixedMu_ = !vm["fixedSignalStrength"].defaulted();
    saveFitResult_ = vm.count("saveFitResult");
}

bool ChannelCompatibilityCheck::runSpecific(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double &limit, double &limitErr, const double *hint) { 

  RooRealVar *r = dynamic_cast<RooRealVar *>(mc_s->GetParametersOfInterest()->first());
  if (fixedMu_) { r->setVal(mu_); r->setConstant(true); }
  else          { r->setVal(preFitValue_); r->setConstant(false); }

  RooSimultaneous *sim = dynamic_cast<RooSimultaneous *>(mc_s->GetPdf());
  if (sim == 0) throw std::logic_error("Cannot use ChannelCompatibilityCheck if the pdf is not a RooSimultaneous");

  RooAbsCategoryLValue *cat = (RooAbsCategoryLValue *) sim->indexCat().Clone();
  int nbins = cat->numBins((const char *)0);
  TString satname = TString::Format("%s_freeform", sim->GetName());
  std::auto_ptr<RooSimultaneous> newsim((typeid(*sim) == typeid(RooSimultaneousOpt)) ? new RooSimultaneousOpt(satname, "", *cat) : new RooSimultaneous(satname, "", *cat)); 
  std::map<std::string,std::string> rs;
  RooArgList minosVars, minosOneVar; if (runMinos_) minosOneVar.add(*r);
  for (int ic = 0, nc = nbins; ic < nc; ++ic) {
      cat->setBin(ic);
      RooAbsPdf *pdfi = sim->getPdf(cat->getLabel());
      if (pdfi == 0) continue;
      RooCustomizer customizer(*pdfi, "freeform");
      TString riName = TString::Format("_ChannelCompatibilityCheck_%s_%s", r->GetName(), nameForLabel(cat->getLabel()).c_str());
      rs.insert(std::pair<std::string,std::string>(nameForLabel(cat->getLabel()), riName.Data()));
      if (w->var(riName) == 0) {
        w->factory(TString::Format("%s[%g,%g]", riName.Data(), r->getMin(), r->getMax()));
      }
      customizer.replaceArg(*r, *w->var(riName));
      newsim->addPdf((RooAbsPdf&)*customizer.build(), cat->getLabel());
      if (runMinos_ && !minosVars.find(riName)) minosVars.add(*w->var(riName));
  }

  CloseCoutSentry sentry(verbose < 2);
  const RooCmdArg &constCmdArg = withSystematics  ? RooFit::Constrain(*mc_s->GetNuisanceParameters()) : RooFit::NumCPU(1); // use something dummy 
  std::auto_ptr<RooFitResult> result_nominal (doFit(   *sim, data, minosOneVar, constCmdArg, runMinos_)); // let's run Hesse if we want to run Minos
  std::auto_ptr<RooFitResult> result_freeform(doFit(*newsim, data, minosVars,   constCmdArg, runMinos_));
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
        if (runMinos_ && do95_) {
            printf("Nominal fit  : %s = %7.4f  %+6.4f/%+6.4f (68%% CL)\n", r->GetName(), rNominal->getVal(), rNominal->getAsymErrorLo(), rNominal->getAsymErrorHi());
            printf("               %s = %7.4f  %+6.4f/%+6.4f (95%% CL)\n", r->GetName(), rNominal->getVal(), rNominal->getMin("err95")-rNominal->getVal(), rNominal->getMax("err95")-rNominal->getVal());
        } else if (runMinos_) {
            printf("Nominal fit  : %s = %7.4f  %+6.4f/%+6.4f\n", r->GetName(), rNominal->getVal(), rNominal->getAsymErrorLo(), rNominal->getAsymErrorHi());
        } else {
            printf("Nominal fit  : %s = %7.4f  +/- %6.4f\n", r->GetName(), rNominal->getVal(), rNominal->getError());
        }
    }
    for (std::map<std::string,std::string>::const_iterator it = rs.begin(), ed = rs.end(); it != ed; ++it) {
        RooRealVar *ri = (RooRealVar*) result_freeform->floatParsFinal().find(it->second.c_str());
        if (runMinos_ && do95_) {
            printf("Alternate fit: %s = %7.4f  %+6.4f/%+6.4f (68%% CL) in channel %s\n", r->GetName(), ri->getVal(), ri->getAsymErrorLo(), ri->getAsymErrorHi(), it->first.c_str());
            printf("               %s = %7.4f  %+6.4f/%+6.4f (95%% CL) in channel %s\n", r->GetName(), ri->getVal(), ri->getMin("err95")-ri->getVal(), ri->getMax("err95")-ri->getVal(), it->first.c_str());
        } else if (runMinos_) {
            printf("Alternate fit: %s = %7.4f  %+6.4f/%+6.4f   in channel %s\n", r->GetName(), ri->getVal(), ri->getAsymErrorLo(), ri->getAsymErrorHi(), it->first.c_str());
        } else {
            printf("Alternate fit: %s = %7.4f  +/- %6.4f   in channel %s\n", r->GetName(), ri->getVal(), ri->getError(), it->first.c_str());
        }
    }
  }
  std::cout << "Chi2-like compatibility variable: " << limit << std::endl;

  if (saveFitResult_) {
      writeToysHere->GetFile()->WriteTObject(result_nominal.release(),  "fit_nominal"  );
      writeToysHere->GetFile()->WriteTObject(result_freeform.release(), "fit_alternate");
  }
  return true;
}

std::string ChannelCompatibilityCheck::nameForLabel(const char *label)
{
    std::string ret(label);
    for (std::vector<std::string>::const_iterator it = groups_.begin(), ed = groups_.end(); it != ed; ++it) {
        if (ret.find(*it) != std::string::npos) { ret = *it; break; }
    }
    return ret;
}

