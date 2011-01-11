#include "HiggsAnalysis/CombinedLimit/interface/MarkovChainMC.h"
#include <stdexcept> 
#include "RooRealVar.h"
#include "RooArgSet.h"
#include "RooUniform.h"
#include "RooWorkspace.h"
#include "RooFitResult.h"
#include "RooStats/MCMCCalculator.h"
#include "RooStats/MCMCInterval.h"
#include "RooStats/ModelConfig.h"
#include "RooStats/ProposalHelper.h"
#include "RooStats/ProposalFunction.h"
#include "RooStats/RooStatsUtils.h"
#include "HiggsAnalysis/CombinedLimit/interface/Combine.h"
#include "HiggsAnalysis/CombinedLimit/interface/TestProposal.h"
#include "HiggsAnalysis/CombinedLimit/interface/DebugProposal.h"
#include "HiggsAnalysis/CombinedLimit/interface/CloseCoutSentry.h"
#include "HiggsAnalysis/CombinedLimit/interface/RooFitGlobalKillSentry.h"

using namespace RooStats;

MarkovChainMC::MarkovChainMC() : 
    LimitAlgo("Markov Chain MC specific options") 
{
    options_.add_options()
        ("iteration,i", boost::program_options::value<unsigned int>(&iterations_)->default_value(20000), "Number of iterations")
        ("tries", boost::program_options::value<unsigned int>(&tries_)->default_value(1), "Number of times to run the MCMC on the same data")
        ("burnInSteps,b", boost::program_options::value<unsigned int>(&burnInSteps_)->default_value(50), "Burn in steps")
        ("nBins,B", boost::program_options::value<unsigned int>(&numberOfBins_)->default_value(1000), "Number of bins")
        ("proposal", boost::program_options::value<std::string>(&proposalTypeName_)->default_value("gaus"), 
                              "Proposal function to use: 'fit', 'uniform', 'gaus'")
        ("runMinos",          "Run MINOS when fitting the data")
        ("noReset",           "Don't reset variable state after fit")
        ("updateProposalParams", 
                boost::program_options::value<bool>(&updateProposalParams_)->default_value(false), 
                "Control ProposalHelper::SetUpdateProposalParameters")
        ("propHelperCacheSize", 
                boost::program_options::value<unsigned int>(&proposalHelperCacheSize_)->default_value(100), 
                "Cache Size for ProposalHelper")
        ("propHelperWidthRangeDivisor", 
                boost::program_options::value<float>(&proposalHelperWidthRangeDivisor_)->default_value(5.), 
                "Sets the fractional size of the gaussians in the proposal")
        ("propHelperUniformFraction", 
                boost::program_options::value<float>(&proposalHelperUniformFraction_)->default_value(0), 
                "Add a fraction of uniform proposals to the algorithm")
        ("debugProposal", boost::program_options::value<int>(&debugProposal_)->default_value(0), "Printout the first N proposals")
        ("cropNSigmas", 
                boost::program_options::value<float>(&cropNSigmas_)->default_value(0),
                "crop range of all parameters to N times their uncertainty") 
    ;
}

void MarkovChainMC::applyOptions(const boost::program_options::variables_map &vm) {
    if      (proposalTypeName_ == "fit")     proposalType_ = FitP;
    else if (proposalTypeName_ == "uniform") proposalType_ = UniformP;
    else if (proposalTypeName_ == "gaus")    proposalType_ = MultiGaussianP;
    else if (proposalTypeName_ == "test")    proposalType_ = TestP;
    else {
        std::cerr << "MarkovChainMC: proposal type " << proposalTypeName_ << " not known." << "\n" << options_ << std::endl;
        throw std::invalid_argument("MarkovChainMC: unsupported proposal");
    }
    if (proposalType_ == MultiGaussianP && !withSystematics) { 
        std::cerr << "Sorry, the multi-gaussian proposal does not work without systematics.\n" << 
                     "Please use the flat proposal instead (--proposal flat)\n" << std::endl;
        throw std::invalid_argument("MarkovChainMC: unsupported proposal");
    }
        
    runMinos_ = vm.count("runMinos");
    noReset_  = vm.count("noReset");
}

bool MarkovChainMC::run(RooWorkspace *w, RooAbsData &data, double &limit, const double *hint) {
  RooFitGlobalKillSentry silence(verbose > 0 ? RooFit::INFO : RooFit::WARNING);

  CloseCoutSentry coutSentry(verbose <= 0); // close standard output and error, so that we don't flood them with minuit messages
  double sum = 0, sum2 = 0, suma = 0; int num = 0;
  for (unsigned int i = 0; i < tries_; ++i) {
      if (int nacc = runOnce(w,data,limit,hint)) {
          ++num;
          sum  += limit;
          sum2 += limit * limit;
          suma += nacc;
      }
  }
  if (num == 0) return false;
  coutSentry.clear();

  if (verbose >= 0) {
      std::cout << "\n -- MarkovChainMC -- " << "\n";
      limit = sum / num;
      suma  = suma / (num * double(iterations_));
      if (num > 1) {
          double limitErr = sqrt((sum2/num-limit*limit)/(num-1));
          std::cout << "Limit: r < " << limit << " +/- " << limitErr << " @ " << cl * 100 << "% CL (" << num << " tries)" << std::endl;
          if (verbose > 0) std::cout << "Average chain acceptance: " << suma << std::endl;
      } else {
          std::cout << "Limit: r < " << limit << " @ " << cl * 100 << "% CL" << std::endl;
      }
  }
  return true;
}
int MarkovChainMC::runOnce(RooWorkspace *w, RooAbsData &data, double &limit, const double *hint) const {
  RooRealVar *r = w->var("r");
  RooArgSet  poi(*r);
  RooArgSet const &obs = *w->set("observables");

  if ((hint != 0) && (*hint > r->getMin())) {
    r->setMax(std::min<double>(5*(*hint), r->getMax()));
  }

  if (withSystematics && (w->set("nuisances") == 0)) {
    throw std::logic_error("MarkovChainMC: running with systematics enabled, but nuisances or nuisancePdf not defined.");
  }
  
  w->loadSnapshot("clean");
  RooAbsPdf *prior = w->pdf("prior"); if (prior == 0) { throw std::logic_error("Missing prior"); }
  std::auto_ptr<RooFitResult> fit(0);
  if (proposalType_ == FitP || (cropNSigmas_ > 0)) {
      CloseCoutSentry coutSentry(verbose <= 1); // close standard output and error, so that we don't flood them with minuit messages
      fit.reset(w->pdf("model_s")->fitTo(data, RooFit::Save(), RooFit::Minos(runMinos_)));
      coutSentry.clear();
      if (fit.get() == 0) { std::cerr << "Fit failed." << std::endl; return false; }
      if (verbose > 1) fit->Print("V");
      if (!noReset_) w->loadSnapshot("clean");
  }

  if (cropNSigmas_ > 0) {
      const RooArgList &fpf = fit->floatParsFinal();
      for (int i = 0, n = fpf.getSize(); i < n; ++i) {
          RooRealVar *fv = dynamic_cast<RooRealVar *>(fpf.at(i));
          if (std::string("r") == fv->GetName()) continue;
          RooRealVar *v  = w->var(fv->GetName());
          double min = v->getMin(), max = v->getMax();
          if (fv->hasAsymError(false)) {
              min = (std::max(v->getMin(), fv->getVal() + cropNSigmas_ * fv->getAsymErrorLo()));
              max = (std::min(v->getMax(), fv->getVal() + cropNSigmas_ * fv->getAsymErrorHi()));
          } else if (fv->hasError(false)) {
              min = (std::max(v->getMin(), fv->getVal() - cropNSigmas_ * fv->getError()));
              max = (std::min(v->getMax(), fv->getVal() + cropNSigmas_ * fv->getError()));
          }
          if (verbose > 1) {
              std::cout << "  " << fv->GetName() << "[" << v->getMin() << ", " << v->getMax() << "] -> [" << min << ", " << max << "]" << std::endl;
          }
          v->setMin(min); v->setMax(max);
      }
  }

  ModelConfig modelConfig("sb_model", w);
  modelConfig.SetPdf(*w->pdf("model_s"));
  modelConfig.SetObservables(obs);
  modelConfig.SetParametersOfInterest(poi);
  if (withSystematics) modelConfig.SetNuisanceParameters(*w->set("nuisances"));
 
  std::auto_ptr<ProposalFunction> ownedPdfProp; 
  ProposalFunction* pdfProp = 0;
  ProposalHelper ph;
  switch (proposalType_) {
    case UniformP:  
        if (verbose) std::cout << "Using uniform proposal" << std::endl;
        ownedPdfProp.reset(new UniformProposal());
        pdfProp = ownedPdfProp.get();
        break;
    case FitP:
        if (verbose) std::cout << "Using fit proposal" << std::endl;
        ph.SetVariables(fit->floatParsFinal());
        ph.SetCovMatrix(fit->covarianceMatrix());
        pdfProp = ph.GetProposalFunction();
        break;
    case MultiGaussianP:
        if (verbose) std::cout << "Using multi-gaussian proposal" << std::endl;
        ph.SetVariables(*w->set("nuisances"));
        ph.SetWidthRangeDivisor(proposalHelperWidthRangeDivisor_);
        pdfProp = ph.GetProposalFunction();
        break;
    case TestP:
        ownedPdfProp.reset(new TestProposal(proposalHelperWidthRangeDivisor_));
        pdfProp = ownedPdfProp.get();
        break;
  }
  if (proposalType_ != UniformP) {
      ph.SetUpdateProposalParameters(updateProposalParams_);
      ph.SetCacheSize(proposalHelperCacheSize_);
      if (proposalHelperUniformFraction_ > 0) ph.SetUniformFraction(proposalHelperUniformFraction_);
  }

  std::auto_ptr<DebugProposal> pdfDebugProp(debugProposal_ > 0 ? new DebugProposal(pdfProp, w->pdf("model_s"), &data, debugProposal_) : 0);
  
  MCMCCalculator mc(data, modelConfig);
  mc.SetNumIters(iterations_); 
  mc.SetConfidenceLevel(cl);
  mc.SetNumBurnInSteps(burnInSteps_); 
  mc.SetProposalFunction(debugProposal_ > 0 ? *pdfDebugProp : *pdfProp);
  mc.SetNumBins (numberOfBins_) ; // bins to use for RooRealVars in histograms
  mc.SetLeftSideTailFraction(0);
  mc.SetPriorPdf(*prior);

  std::auto_ptr<MCMCInterval> mcInt;
  try {  
      mcInt.reset((MCMCInterval*)mc.GetInterval()); 
  } catch (std::length_error &ex) {
      mcInt.reset(0);
  }
  if (mcInt.get() == 0) return false;
  limit = mcInt->UpperLimit(*r);

  return mcInt->GetChain()->Size();
}

