#include "Mixing/Base/interface/PileUp.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/InputSourceDescription.h"
#include "FWCore/Sources/interface/VectorInputSourceFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "CLHEP/Random/RandPoissonQ.h"
#include "CLHEP/Random/RandFlat.h"

#include "TRandom.h"
#include "TFile.h"
#include "TH1F.h"

#include <algorithm>

namespace edm {
  PileUp::PileUp(ParameterSet const& pset, int const minb, int const maxb, double averageNumber, TH1F * const histo, const bool playback) :
    type_(pset.getParameter<std::string>("type")),
    minBunch_(minb),
    maxBunch_(maxb),
    averageNumber_(averageNumber),
    intAverage_(static_cast<int>(averageNumber)),
    histo_(histo),
    histoDistribution_(type_ == "histo"),
    probFunctionDistribution_(type_ == "probFunction"),
    poisson_(type_ == "poisson"),
    fixed_(type_ == "fixed"),
    none_(type_ == "none"),
    input_(VectorInputSourceFactory::get()->makeVectorInputSource(pset, InputSourceDescription()).release()),
    poissonDistribution_(0),
    playback_(playback),
    sequential_(pset.getUntrackedParameter<bool>("sequential", false)),
    seed_(pset.getParameter<edm::ParameterSet>("nbPileupEvents").getUntrackedParameter<int>("seed",1234))
   {

    
    edm::Service<edm::RandomNumberGenerator> rng;
    if (!rng.isAvailable()) {
      throw cms::Exception("Configuration")
	<< "PileUp requires the RandomNumberGeneratorService\n"
	"which is not present in the configuration file.  You must add the service\n"
	"in the configuration file or remove the modules that require it.";
    }
    
    CLHEP::HepRandomEngine& engine = rng->getEngine();
    poissonDistribution_ = new CLHEP::RandPoissonQ(engine, averageNumber_);
    
    // Get seed for the case when using user histogram or probability function
    if (histoDistribution_ || probFunctionDistribution_){ 
      gRandom->SetSeed(seed_);
      LogInfo("MixingModule") << " Change seed for " << type_ << " mode. The seed is set to " << seed_;
    } 
     
        
    if (!(histoDistribution_ || probFunctionDistribution_ || poisson_ || fixed_ || none_)) {
      throw cms::Exception("Illegal parameter value","PileUp::PileUp(ParameterSet const& pset)")
        << "'type' parameter (a string) has a value of '" << type_ << "'.\n"
        << "Legal values are 'poisson', 'fixed', or 'none'\n";
    }

    manage_OOT_ = pset.getUntrackedParameter<bool>("manage_OOT", false);

    if(manage_OOT_) { // figure out what the parameters are

      if (playback_) throw cms::Exception("Illegal parameter clash","PileUp::PileUp(ParameterSet const& pset)")
	<< " manage_OOT option not allowed with playback ";

      std::string OOT_type = pset.getUntrackedParameter<std::string>("OOT_type");

      if(OOT_type == "Poisson" || OOT_type == "poisson") {
	poisson_OOT_ = true;
	poissonDistr_OOT_ = new CLHEP::RandPoisson(engine);
      }
      else if(OOT_type == "Fixed" || OOT_type == "fixed") {
	fixed_OOT_ = true;
	// read back the fixed number requested out-of-time
	intFixed_OOT_ = pset.getUntrackedParameter<int>("intFixed_OOT", -1);
	if(intFixed_OOT_ < 0) {
	  throw cms::Exception("Illegal parameter value","PileUp::PileUp(ParameterSet const& pset)") 
	    << " Fixed out-of-time pileup requested, but no fixed value given ";
	}
      }
      else {
	throw cms::Exception("Illegal parameter value","PileUp::PileUp(ParameterSet const& pset)")
	  << "'OOT_type' parameter (a string) has a value of '" << OOT_type << "'.\n"
	  << "Legal values are 'poisson' or 'fixed'\n";
      }
      edm::LogInfo("MixingModule") <<" Out-of-time pileup will be generated with a " << OOT_type << " distribution. " ;
    }
    
  }

  PileUp::~PileUp() {
    delete poissonDistribution_;
  }

  void
  PileUp::readPileUp(std::vector<EventPrincipalVector> & result,std::vector<std::vector<edm::EventID> > &ids) {

    // set up vector of event counts for each bunch crossing ahead of time, so that we can
    // allow for an arbitrary distribution for out-of-time vs. in-time pileup

    std::vector<int> nint;

    // if we are managing the distribution of out-of-time pileup separately, select the distribution for bunch
    // crossing zero first, save it for later.

    int nzero_crossing = -1;

    if(manage_OOT_) {
      if (none_){
	nzero_crossing = 0;
      }else if (poisson_){
	nzero_crossing =  poissonDistribution_->fire() ;
      }else if (fixed_){
	nzero_crossing =  intAverage_ ;
      }else if (histoDistribution_ || probFunctionDistribution_){
	double d = histo_->GetRandom();
	//n = (int) floor(d + 0.5);  // incorrect for bins with integer edges
	nzero_crossing =  int(d);
      }
    }
            
    for (int i = minBunch_; i <= maxBunch_; ++i) {
      
      if (playback_){
	nint.push_back( ids[i-minBunch_].size() );
      //} else if (sequential_) {  // just read many sequentially... why specify?
      // For now, the use case for sequential read reads only one event at a time.
      // nint.push_back( 1 );
      } 
      else if(manage_OOT_) {
	if(i==0 && !poisson_OOT_) nint.push_back(nzero_crossing);
	else{
	  if(poisson_OOT_) {
	    nint.push_back( poissonDistr_OOT_->fire(float(nzero_crossing)) );
	  }
	  else {
	    nint.push_back( intFixed_OOT_ );
	  }	  
	}
      } 
      else {	
	if (none_){
	  nint.push_back(0);
	}else if (poisson_){
	  nint.push_back( poissonDistribution_->fire() );
	}else if (fixed_){
	  nint.push_back( intAverage_ );
	}else if (histoDistribution_ || probFunctionDistribution_){
	  double d = histo_->GetRandom();
	  //n = (int) floor(d + 0.5);  // incorrect for bins with integer edges
	  nint.push_back( int(d) );
	}

      }
    }

    int n=0;
      
    for (int i = minBunch_; i <= maxBunch_; ++i) {
      EventPrincipalVector eventVector;

      n = nint[i-minBunch_];

      eventVector.reserve(n);
      while (n > 0) {
        EventPrincipalVector oneResult;
        oneResult.reserve(n);
	std::vector<edm::EventID> oneResultPlayback;
	oneResultPlayback.reserve(n);
	if (playback_)   {
	  input_->readManySpecified(ids[i-minBunch_],oneResult);  // playback
	} else if (sequential_) {
	  unsigned int file;
	  input_->readManySequential(n, oneResult, file);  // sequential
	  for (int j=0;j<(int)oneResult.size();j++){
	    oneResultPlayback.push_back(oneResult[j]->id());
	  }
	  ids[i-minBunch_] = oneResultPlayback;
	} else  {
	  unsigned int file;   //FIXME: need unsigned filenr?
	  input_->readManyRandom(n, oneResult,file);     //no playback
	  for (int j=0;j<(int)oneResult.size();j++){
	    oneResultPlayback.push_back(oneResult[j]->id());
	  }
	  ids[i-minBunch_] = oneResultPlayback;
	}
        LogDebug("readPileup") << "READ: " << oneResult.size();
        std::copy(oneResult.begin(), oneResult.end(), std::back_inserter(eventVector));
	n -= oneResult.size();
      }
      result.push_back(eventVector);
    }
  }

} //namespace edm
