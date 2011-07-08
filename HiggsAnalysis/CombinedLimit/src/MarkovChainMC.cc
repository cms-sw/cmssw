#include "../interface/MarkovChainMC.h"
#include <stdexcept> 
#include <cmath> 
#include "TKey.h"
#include "RooRealVar.h"
#include "RooArgSet.h"
#include "RooUniform.h"
#include "RooWorkspace.h"
#include "RooFitResult.h"
#include "RooRandom.h"
#include "RooStats/MCMCCalculator.h"
#include "RooStats/MCMCInterval.h"
#include "RooStats/ModelConfig.h"
#include "RooStats/ProposalHelper.h"
#include "RooStats/ProposalFunction.h"
#include "RooStats/RooStatsUtils.h"
#include "../interface/Combine.h"
#include "../interface/TestProposal.h"
#include "../interface/DebugProposal.h"
#include "../interface/CloseCoutSentry.h"
#include "../interface/RooFitGlobalKillSentry.h"
#include "../interface/JacknifeQuantile.h"

using namespace RooStats;

std::string MarkovChainMC::proposalTypeName_ = "ortho";
MarkovChainMC::ProposalType MarkovChainMC::proposalType_ = TestP;
bool MarkovChainMC::runMinos_ = false;
bool MarkovChainMC::noReset_ = false;
bool MarkovChainMC::updateProposalParams_ = false;
bool MarkovChainMC::updateHint_ = false;
unsigned int MarkovChainMC::iterations_ = 10000;
unsigned int MarkovChainMC::burnInSteps_ = 200;
unsigned int MarkovChainMC::tries_ = 10;
float MarkovChainMC::truncatedMeanFraction_ = 0.0;
bool MarkovChainMC::adaptiveTruncation_ = true;
float MarkovChainMC::hintSafetyFactor_ = 5.;
bool MarkovChainMC::saveChain_ = false;
bool MarkovChainMC::mergeChains_ = false;
bool MarkovChainMC::readChains_ = false;
float MarkovChainMC::proposalHelperWidthRangeDivisor_ = 5.;
float MarkovChainMC::proposalHelperUniformFraction_ = 0.0;
bool  MarkovChainMC::alwaysStepPoi_ = false;
float MarkovChainMC::cropNSigmas_ = 0;
int   MarkovChainMC::debugProposal_ = false;

MarkovChainMC::MarkovChainMC() : 
    LimitAlgo("Markov Chain MC specific options") 
{
    options_.add_options()
        ("iteration,i", boost::program_options::value<unsigned int>(&iterations_)->default_value(iterations_), "Number of iterations")
        ("tries", boost::program_options::value<unsigned int>(&tries_)->default_value(tries_), "Number of times to run the MCMC on the same data")
        ("burnInSteps,b", boost::program_options::value<unsigned int>(&burnInSteps_)->default_value(burnInSteps_), "Burn in steps")
        ("proposal", boost::program_options::value<std::string>(&proposalTypeName_)->default_value(proposalTypeName_), 
                              "Proposal function to use: 'fit', 'uniform', 'gaus', 'ortho' (also known as 'test')")
        ("runMinos",          "Run MINOS when fitting the data")
        ("noReset",           "Don't reset variable state after fit")
        ("updateHint",        "Update hint with the results")
        ("updateProposalParams", 
                boost::program_options::value<bool>(&updateProposalParams_)->default_value(updateProposalParams_), 
                "Control ProposalHelper::SetUpdateProposalParameters")
        ("propHelperWidthRangeDivisor", 
                boost::program_options::value<float>(&proposalHelperWidthRangeDivisor_)->default_value(proposalHelperWidthRangeDivisor_), 
                "Sets the fractional size of the gaussians in the proposal")
        ("alwaysStepPOI", boost::program_options::value<bool>(&alwaysStepPoi_)->default_value(alwaysStepPoi_),
                            "When using 'ortho' proposal, always step also the parameter of interest")
        ("propHelperUniformFraction", 
                boost::program_options::value<float>(&proposalHelperUniformFraction_)->default_value(proposalHelperUniformFraction_), 
                "Add a fraction of uniform proposals to the algorithm")
        ("debugProposal", boost::program_options::value<int>(&debugProposal_)->default_value(debugProposal_), "Printout the first N proposals")
        ("cropNSigmas", 
                boost::program_options::value<float>(&cropNSigmas_)->default_value(cropNSigmas_),
                "crop range of all parameters to N times their uncertainty") 
        ("truncatedMeanFraction", 
                boost::program_options::value<float>(&truncatedMeanFraction_)->default_value(truncatedMeanFraction_), 
                "Discard this fraction of the results before computing the mean and rms")
        ("adaptiveTruncation", boost::program_options::value<bool>(&adaptiveTruncation_)->default_value(adaptiveTruncation_),
                            "When averaging multiple runs, ignore results that are more far away from the median than the inter-quartile range")
        ("hintSafetyFactor",
                boost::program_options::value<float>(&hintSafetyFactor_)->default_value(hintSafetyFactor_),
                "set range of integration equal to this number of times the hinted limit")
        ("saveChain", "Save MarkovChain to output file")
        ("mergeChains", "Merge MarkovChains instead of averaging limits")
        ("readChains", "Just read MarkovChains from toysFile instead of running MCMC directly")
    ;
}

void MarkovChainMC::applyOptions(const boost::program_options::variables_map &vm) {
    if      (proposalTypeName_ == "fit")     proposalType_ = FitP;
    else if (proposalTypeName_ == "uniform") proposalType_ = UniformP;
    else if (proposalTypeName_ == "gaus")    proposalType_ = MultiGaussianP;
    else if (proposalTypeName_ == "ortho")   proposalType_ = TestP;
    else if (proposalTypeName_ == "test")    proposalType_ = TestP;
    else {
        std::cerr << "MarkovChainMC: proposal type " << proposalTypeName_ << " not known." << "\n" << options_ << std::endl;
        throw std::invalid_argument("MarkovChainMC: unsupported proposal");
    }
        
    runMinos_ = vm.count("runMinos");
    noReset_  = vm.count("noReset");
    updateHint_  = vm.count("updateHint");

    mass_ = vm["mass"].as<float>();
    saveChain_   = vm.count("saveChain");
    mergeChains_ = vm.count("mergeChains");
    readChains_  = vm.count("readChains");

    if (mergeChains_ && !saveChain_ && !readChains_) chains_.SetOwner(true);
}

bool MarkovChainMC::run(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double &limit, double &limitErr, const double *hint) {
  if (proposalType_ == MultiGaussianP && !withSystematics) { 
      std::cerr << "Sorry, the multi-gaussian proposal does not work without systematics.\n" << 
                   "Uniform proposal will be used instead.\n" << std::endl;
      proposalType_ = UniformP;
  }

  RooFitGlobalKillSentry silence(verbose > 0 ? RooFit::INFO : RooFit::WARNING);

  CloseCoutSentry coutSentry(verbose <= 0); // close standard output and error, so that we don't flood them with minuit messages

  double suma = 0; int num = 0;
  double savhint = (hint ? *hint : -1); const double *thehint = hint;
  std::vector<double> limits;
  if (readChains_)  {
      readChains(*mc_s->GetParametersOfInterest(), limits);
  } else {
      for (unsigned int i = 0; i < tries_; ++i) {
          if (int nacc = runOnce(w,mc_s,mc_b,data,limit,limitErr,thehint)) {
              suma += nacc;
              if (verbose > 1) std::cout << "Limit from this run: " << limit << std::endl;
              limits.push_back(limit);
              if (updateHint_ && tries_ > 1 && limit > savhint) { 
                if (verbose > 0) std::cout << "Updating hint from " << savhint << " to " << limit << std::endl;
                savhint = limit; thehint = &savhint; 
              }
          }
      }
  } 
  num = limits.size();
  if (num == 0) return false;
  // average acceptance
  suma  = suma / (num * double(iterations_));
  limitAndError(limit, limitErr, limits);
  if (mergeChains_) {
    std::cout << "Limit from averaging:    " << limit << " +/- " << limitErr << std::endl;
    // copy constructors don't work, so we just have to leak memory :-(
    RooStats::MarkovChain *merged = mergeChains();
    limitFromChain(limit, limitErr, *mc_s->GetParametersOfInterest(), *merged);
    std::cout << "Limit from merged chain: " << limit << " +/- " << limitErr << std::endl;
  }
  coutSentry.clear();

  if (verbose >= 0) {
      std::cout << "\n -- MarkovChainMC -- " << "\n";
      RooRealVar *r = dynamic_cast<RooRealVar *>(mc_s->GetParametersOfInterest()->first());
      if (num > 1) {
          std::cout << "Limit: " << r->GetName() <<" < " << limit << " +/- " << limitErr << " @ " << cl * 100 << "% CL (" << num << " tries)" << std::endl;
          if (verbose > 0 && !readChains_) std::cout << "Average chain acceptance: " << suma << std::endl;
      } else {
          std::cout << "Limit: " << r->GetName() <<" < " << limit << " @ " << cl * 100 << "% CL" << std::endl;
      }
  }
  return true;
}
int MarkovChainMC::runOnce(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double &limit, double &limitErr, const double *hint) const {
  RooRealVar *r = dynamic_cast<RooRealVar *>(mc_s->GetParametersOfInterest()->first());

  if ((hint != 0) && (*hint > r->getMin())) {
    r->setMax(hintSafetyFactor_*(*hint));
  }

  if (withSystematics && (mc_s->GetNuisanceParameters() == 0)) {
    throw std::logic_error("MarkovChainMC: running with systematics enabled, but nuisances not defined.");
  }
  
  w->loadSnapshot("clean");
  std::auto_ptr<RooFitResult> fit(0);
  if (proposalType_ == FitP || (cropNSigmas_ > 0)) {
      CloseCoutSentry coutSentry(verbose <= 1); // close standard output and error, so that we don't flood them with minuit messages
      fit.reset(mc_s->GetPdf()->fitTo(data, RooFit::Save(), RooFit::Minos(runMinos_)));
      coutSentry.clear();
      if (fit.get() == 0) { std::cerr << "Fit failed." << std::endl; return false; }
      if (verbose > 1) fit->Print("V");
      if (!noReset_) w->loadSnapshot("clean");
  }

  if (cropNSigmas_ > 0) {
      const RooArgList &fpf = fit->floatParsFinal();
      for (int i = 0, n = fpf.getSize(); i < n; ++i) {
          RooRealVar *fv = dynamic_cast<RooRealVar *>(fpf.at(i));
          if (std::string(r->GetName()) == fv->GetName()) continue;
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
        ph.SetVariables(*mc_s->GetNuisanceParameters());
        ph.SetWidthRangeDivisor(proposalHelperWidthRangeDivisor_);
        pdfProp = ph.GetProposalFunction();
        break;
    case TestP:
        ownedPdfProp.reset(new TestProposal(proposalHelperWidthRangeDivisor_, alwaysStepPoi_ ? r : NULL));
        pdfProp = ownedPdfProp.get();
        break;
  }
  if (proposalType_ != UniformP) {
      ph.SetUpdateProposalParameters(updateProposalParams_);
      if (proposalHelperUniformFraction_ > 0) ph.SetUniformFraction(proposalHelperUniformFraction_);
  }

  std::auto_ptr<DebugProposal> pdfDebugProp(debugProposal_ > 0 ? new DebugProposal(pdfProp, mc_s->GetPdf(), &data, debugProposal_) : 0);
  
  MCMCCalculator mc(data, *mc_s);
  mc.SetNumIters(iterations_); 
  mc.SetConfidenceLevel(cl);
  mc.SetNumBurnInSteps(burnInSteps_); 
  mc.SetProposalFunction(debugProposal_ > 0 ? *pdfDebugProp : *pdfProp);
  mc.SetLeftSideTailFraction(0);

  if (typeid(*mc_s->GetPriorPdf()) == typeid(RooUniform)) {
    mc.SetPriorPdf(*((RooAbsPdf *)0));
  }

  std::auto_ptr<MCMCInterval> mcInt;
  try {  
      mcInt.reset((MCMCInterval*)mc.GetInterval()); 
  } catch (std::length_error &ex) {
      mcInt.reset(0);
  }
  if (mcInt.get() == 0) return false;
  limit = mcInt->UpperLimit(*r);

  if (saveChain_ || mergeChains_) {
      // Copy-constructors don't work properly, so we just have to leak memory.
      //RooStats::MarkovChain *chain = new RooStats::MarkovChain(*mcInt->GetChain());
      RooStats::MarkovChain *chain = const_cast<RooStats::MarkovChain *>(mcInt.release()->GetChain());
      if (mergeChains_) chains_.Add(chain);
      if (saveChain_)  writeToysHere->WriteTObject(chain,  TString::Format("MarkovChain_mh%g_%u",mass_, RooRandom::integer(std::numeric_limits<UInt_t>::max() - 1)));
      return chain->Size();
  } else {
    return mcInt->GetChain()->Size();
  }
}

void MarkovChainMC::limitAndError(double &limit, double &limitErr, std::vector<double> &limits) const {
  int num = limits.size();
  // possibly remove outliers before computing mean
  if (adaptiveTruncation_ && num >= 10) {
      std::sort(limits.begin(), limits.end());
      // determine location and size of the sample
      double median = (num % 2 ? limits[num/2] : 0.5*(limits[num/2] + limits[num/2+1]));
      double iqr = limits[3*num/4] - limits[num/4];
      // determine range of plausible values
      double min = median - iqr, max = median + iqr; 
      int start = 0, end = num-1; 
      while (start < end   && limits[start] < min) ++start;
      while (end   > start && limits[end]   > max) --end;
      num = end-start+1;
      // compute mean and rms of accepted part
      limit = 0; limitErr = 0;
      for (int k = start; k <= end; ++k) limit += limits[k];
      limit /= num;
      for (int k = start; k <= end; k++) limitErr += (limits[k]-limit)*(limits[k]-limit);
      limitErr = (num > 1 ? sqrt(limitErr/(num*(num-1))) : 0);
      std::cout << "Result from truncated mean: " << limit << " +/- " << limitErr << std::endl;
#if 0
      QuantileCalculator qc(limits);
      std::pair<double,double> qn = qc.quantileAndError(0.5, QuantileCalculator::Simple);
      std::pair<double,double> qs = qc.quantileAndError(0.5, QuantileCalculator::Sectioning);
      std::pair<double,double> qj = qc.quantileAndError(0.5, QuantileCalculator::Jacknife);
      std::cout << "Median of limits (simple):     " << qn.first << " +/- " << qn.second << std::endl;
      std::cout << "Median of limits (sectioning): " << qs.first << " +/- " << qs.second << std::endl;
      std::cout << "Median of limits (jacknife):   " << qj.first << " +/- " << qj.second << std::endl;
#endif
  } else {
      int noutl = floor(truncatedMeanFraction_ * num);
      if (noutl >= 1) { 
          std::sort(limits.begin(), limits.end());
          double median = (num % 2 ? limits[num/2] : 0.5*(limits[num/2] + limits[num/2+1]));
          for (int k = 0; k < noutl; ++k) {
              // make sure outliers are all at the end
              if (std::abs(limits[0]-median) > std::abs(limits[num-k-1]-median)) {
                  std::swap(limits[0], limits[num-k-1]);
              }
          }
          num -= noutl;
      }
      // compute mean and rms
      limit = 0; limitErr = 0;
      for (int k = 0; k < num; k++) limit += limits[k];
      limit /= num;
      for (int k = 0; k < num; k++) limitErr += (limits[k]-limit)*(limits[k]-limit);
      limitErr = (num > 1 ? sqrt(limitErr/(num*(num-1))) : 0);
  }
}

RooStats::MarkovChain *MarkovChainMC::mergeChains() const
{
    if (chains_.GetSize() == 0) throw std::runtime_error("No chains to merge");
    if (verbose > 1) std::cout << "Will merge " << chains_.GetSize() << " chains." << std::endl;
    TIter iter(&chains_);
    RooStats::MarkovChain *merged = new RooStats::MarkovChain((RooStats::MarkovChain &)*iter.Next());
    if (verbose > 1) std::cout << "Staring from the first chain, of " << merged->Size() << " entries" << std::endl;
    for (RooStats::MarkovChain *other = (RooStats::MarkovChain *) iter.Next();
         other != 0;
         other = (RooStats::MarkovChain *) iter.Next()) {
        if (verbose > 1) std::cout << "Then adding chain of " << other->Size() << " entries skipping the first " <<  burnInSteps_ << std::endl;
        for (int i = burnInSteps_, n = other->Size(); i < n; ++i) {
            const RooArgSet *point = other->Get(i);
            double nllval = other->NLL();
            double weight = other->Weight();
            merged->Add(const_cast<RooArgSet&>(*point),nllval,weight);
            if (verbose > 1 && (i % 500 == 0)) std::cout << "   added " << i << "/" << other->Size() << " entries." << std::endl;
        }
    }
    return merged;
}

void MarkovChainMC::readChains(const RooArgSet &poi, std::vector<double> &limits)
{
    double mylim, myerr;
    chains_.Clear();
    chains_.SetOwner(false);
    if (!readToysFromHere) throw std::logic_error("Cannot use readChains: option toysFile not specified, or input file empty");
    TDirectory *toyDir = readToysFromHere->GetDirectory("toys");
    if (!toyDir) throw std::logic_error("Cannot use readChains: empty toy dir in input file empty");
    TString prefix = TString::Format("MarkovChain_mh%g_",mass_);
    TIter next(toyDir->GetListOfKeys()); TKey *k;
    while ((k = (TKey *) next()) != 0) {
        if (TString(k->GetName()).Index(prefix) != 0) continue;
        RooStats::MarkovChain *toy = dynamic_cast<RooStats::MarkovChain *>(toyDir->Get(k->GetName()));
        if (toy == 0) continue;
        limitFromChain(mylim, myerr, poi, *toy);
        chains_.Add(toy);
        limits.push_back(mylim);
    }
    if (verbose) { std::cout << "Read " << chains_.GetSize() << " Markov Chains from input file." << std::endl; }
}

void 
MarkovChainMC::limitFromChain(double &limit, double &limitErr, const RooArgSet &poi, RooStats::MarkovChain &chain) 
{
    MCMCInterval interval("",poi,chain);
    RooArgList axes(poi);
    interval.SetConfidenceLevel(cl);
    interval.SetIntervalType(MCMCInterval::kTailFraction);
    interval.SetLeftSideTailFraction(0);
    interval.SetNumBurnInSteps(burnInSteps_);
    interval.SetAxes(axes);
    limit = interval.UpperLimit((RooRealVar&)*poi.first());
#if 0
    std::cout << "Limit from chain (naive):    " << limit << std::endl;
    QuantileCalculator qc(*chain.GetAsConstDataSet(), poi.first()->GetName(), burnInSteps_);
    qc.randomizePoints();
    std::pair<double,double> qn = qc.quantileAndError(0.95, QuantileCalculator::Simple);
    std::pair<double,double> qs = qc.quantileAndError(0.95, QuantileCalculator::Sectioning);
    std::pair<double,double> qj = qc.quantileAndError(0.95, QuantileCalculator::Jacknife);
    std::cout << "Limit from chain (simple):     " << qn.first << " +/- " << qn.second << std::endl;
    std::cout << "Limit from chain (sectioning): " << qs.first << " +/- " << qs.second << std::endl;
    std::cout << "Limit from chain (jacknife):   " << qj.first << " +/- " << qj.second << std::endl;
#endif
}


