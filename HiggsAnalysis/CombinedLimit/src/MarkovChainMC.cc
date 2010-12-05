#include "HiggsAnalysis/CombinedLimit/interface/MarkovChainMC.h"
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
#include "HiggsAnalysis/CombinedLimit/interface/Combine.h"
#include "HiggsAnalysis/CombinedLimit/interface/TestProposal.h"

using namespace RooStats;

MarkovChainMC::MarkovChainMC() : 
    LimitAlgo("Markov Chain MC specific options") 
{
    options_.add_options()
        ("iteration,i", boost::program_options::value<unsigned int>(&iterations_)->default_value(20000), "Number of iterations")
        ("burnInSteps,b", boost::program_options::value<unsigned int>(&burnInSteps_)->default_value(50), "Burn in steps")
        ("nBins,B", boost::program_options::value<unsigned int>(&numberOfBins_)->default_value(1000), "Number of bins")
        ("proposal,p", boost::program_options::value<std::string>(&proposalTypeName_)->default_value("gaus"), 
                              "Proposal function to use: 'fit', 'uniform', 'gaus'")
        ("runMinos",          "Run MINOS when fitting the data")
        ("noReset",           "Don't reset variable state after fit")
        ("updateProposalParams", 
                boost::program_options::value<bool>(&updateProposalParams_)->default_value(false), 
                "Control ProposalHelper::SetUpdateProposalParameters")
        ("proposalHelperCacheSize", 
                boost::program_options::value<unsigned int>(&proposalHelperCacheSize_)->default_value(100), 
                "Cache Size for ProposalHelper")
        ("proposalHelperWidthRangeDivisor", 
                boost::program_options::value<float>(&proposalHelperWidthRangeDivisor_)->default_value(5.), 
                "Sets the fractional size of the gaussians in the proposal")
        ("proposalHelperUniformFraction", 
                boost::program_options::value<float>(&proposalHelperUniformFraction_)->default_value(0), 
                "Add a fraction of uniform proposals to the algorithm")
    ;
}

void MarkovChainMC::applyOptions(const boost::program_options::variables_map &vm) {
    if      (proposalTypeName_ == "fit")     proposalType_ = FitP;
    else if (proposalTypeName_ == "uniform") proposalType_ = UniformP;
    else if (proposalTypeName_ == "gaus")    proposalType_ = MultiGaussianP;
    else if (proposalTypeName_ == "test")    proposalType_ = TestP;
    else {
        std::cerr << "ERROR: MarkovChainMC: proposal type " << proposalTypeName_ << " not known." << "\n" << options_ << std::endl;
        abort();
    }
        
    runMinos_ = vm.count("runMinos");
    noReset_  = vm.count("noReset");
}
bool MarkovChainMC::run(RooWorkspace *w, RooAbsData &data, double &limit, const double *hint) {
  RooRealVar *r = w->var("r");
  RooArgSet  poi(*r);
  RooArgSet const &obs = *w->set("observables");

  if ((hint != 0) && (*hint > r->getMin())) {
    r->setMax(std::min<double>(3*(*hint), r->getMax()));
  }

  if (withSystematics && (w->set("nuisances") == 0)) {
    std::cerr << "ERROR: nuisances not set. Perhaps you wanted to run with no systematics?\n" << std::endl;
    abort();
  }
  
  w->loadSnapshot("clean");
  RooUniform  flatPrior("flatPrior","flatPrior",*r);
  std::auto_ptr<RooFitResult> fit(0);
  if (proposalType_ == FitP) {
      fit.reset(w->pdf("model_s")->fitTo(data, RooFit::Save(), RooFit::Minos(runMinos_)));
      if (fit.get() == 0) { std::cerr << "Fit failed." << std::endl; return false; }
      if (verbose > 1) fit->Print("V");
      if (!noReset_) w->loadSnapshot("clean");
  }

  ModelConfig modelConfig("sb_model", w);
  modelConfig.SetPdf(*w->pdf("model_s"));
  modelConfig.SetObservables(obs);
  modelConfig.SetParametersOfInterest(poi);
  if (withSystematics) modelConfig.SetNuisanceParameters(*w->set("nuisances"));
  
  ProposalFunction* pdfProp = 0;
  ProposalHelper ph;
  switch (proposalType_) {
    case UniformP:  
        if (verbose) std::cout << "Using uniform proposal" << std::endl;
        pdfProp = new UniformProposal();
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
        pdfProp = new TestProposal();
        proposalType_ = UniformP; // then behave as if it were uniform
        break;
  }
  if (proposalType_ != UniformP) {
      ph.SetUpdateProposalParameters(updateProposalParams_);
      ph.SetCacheSize(proposalHelperCacheSize_);
      if (proposalHelperUniformFraction_ > 0) ph.SetUniformFraction(proposalHelperUniformFraction_);
  }
  
  MCMCCalculator mc(data, modelConfig);
  mc.SetNumIters(iterations_); 
  mc.SetConfidenceLevel(cl);
  mc.SetNumBurnInSteps(burnInSteps_); 
  mc.SetProposalFunction(*pdfProp);
  mc.SetNumBins (numberOfBins_) ; // bins to use for RooRealVars in histograms
  mc.SetLeftSideTailFraction(0);
  mc.SetPriorPdf(flatPrior);
  
  std::auto_ptr<MCMCInterval> mcInt((MCMCInterval*)mc.GetInterval()); 
  if (proposalType_ == UniformP) delete pdfProp; // unfortunately, it looks like the ProposalHelper owns its proposal
  if (mcInt.get() == 0) return false;
  limit = mcInt->UpperLimit(*r);
  if (verbose > 0) {
    std::cout << "\n -- MCMC, flat prior -- " << "\n";
    std::cout << "Limit: r < " << limit << " @ " << cl * 100 << "% CL" << std::endl;

    if (verbose > 1) std::cout << "Getting uncertainty for TF interval" << std::endl;
    if (verbose > 1) printf("  CL %7.5f Limit %9.5f\n", cl, limit);
    double clLo, clHi;
    for (clHi = 1.0; clHi - cl > 0.0005; clHi = 0.5*(cl+clHi)) {
        mcInt->SetConfidenceLevel(clHi);
        double xlimit = mcInt->UpperLimitTailFraction(*r);
        if (verbose > 1) printf("  CL %7.5f Limit %9.5f\n", clHi, xlimit);
        if (xlimit == limit) { 
            clHi += (clHi - cl);
            if (verbose > 1) std::cout << " ... found boundary\n"; 
            break; 
        }
    }
    for (clLo = 0.0; cl - clLo > 0.0005; clLo = 0.5*(cl+clLo)) {
        mcInt->SetConfidenceLevel(clLo);
        double xlimit = mcInt->UpperLimitTailFraction(*r);
        if (verbose > 1) printf("  CL %7.5f Limit %9.5f\n", clLo, xlimit);
        if (xlimit == limit) { 
            if (verbose > 1) std::cout << " ... found boundary\n"; 
            clLo -= (cl - clLo);
            break; 
        }
    }
    mcInt->SetConfidenceLevel(clHi); 
    double limHi = mcInt->UpperLimitTailFraction(*r);
    mcInt->SetConfidenceLevel(clLo); 
    double limLo = mcInt->UpperLimitTailFraction(*r);
        
    std::cout << "       r < " << limLo << " @ " << clLo * 100 << "% CL" << std::endl;
    std::cout << "       r < " << limHi << " @ " << clHi * 100 << "% CL" << std::endl;
    std::cout << "       unertainty: " << 0.5*(limHi-limLo) << std::endl;
  }
  return true;
}
