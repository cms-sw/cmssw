#include "Mixing/Base/interface/PileUp.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/InputSourceDescription.h"
#include "FWCore/Sources/interface/VectorInputSourceFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include <algorithm>

namespace edm {
  PileUp::PileUp(ParameterSet const& pset, double averageNumber, TH1F * const histo, const bool playback) :
    type_(pset.getParameter<std::string>("type")),
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
    poissonDistr_OOT_(0),
    playback_(playback),
    sequential_(pset.getUntrackedParameter<bool>("sequential", false)),
    seed_(pset.getParameter<edm::ParameterSet>("nbPileupEvents").getUntrackedParameter<int>("seed",0))
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
      if(seed_ !=0) {
	gRandom->SetSeed(seed_);
	LogInfo("MixingModule") << " Change seed for " << type_ << " mode. The seed is set to " << seed_;
      }
      else {
	gRandom->SetSeed(engine.getSeed());
      }
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
    delete poissonDistr_OOT_ ;
  }

  void PileUp::CalculatePileup(int MinBunch, int MaxBunch, std::vector<int>& PileupSelection, std::vector<float>& TrueNumInteractions) {

    // if we are managing the distribution of out-of-time pileup separately, select the distribution for bunch
    // crossing zero first, save it for later.

    int nzero_crossing = -1;
    double Fnzero_crossing = -1;

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
	Fnzero_crossing =  d;
      }

    }

    for(int bx = MinBunch; bx < MaxBunch+1; ++bx) {

      if(manage_OOT_) {
	if(bx==0 && !poisson_OOT_) { 
	  PileupSelection.push_back(nzero_crossing) ;
	  TrueNumInteractions.push_back( nzero_crossing );
	}
	else{
	  if(poisson_OOT_) {
	    PileupSelection.push_back(poissonDistr_OOT_->fire(Fnzero_crossing)) ;
	    TrueNumInteractions.push_back( Fnzero_crossing );
	  }
	  else {
	    PileupSelection.push_back(intFixed_OOT_) ;
	    TrueNumInteractions.push_back( intFixed_OOT_ );
	  }  
	}
      }
      else {
	if (none_){
	  PileupSelection.push_back(0);
	  TrueNumInteractions.push_back( 0. );
	}else if (poisson_){
	  PileupSelection.push_back(poissonDistribution_->fire());
	  TrueNumInteractions.push_back( averageNumber_ );
	}else if (fixed_){
	  PileupSelection.push_back(intAverage_);
	  TrueNumInteractions.push_back( intAverage_ );
	}else if (histoDistribution_ || probFunctionDistribution_){
	  double d = histo_->GetRandom();
	  PileupSelection.push_back(int(d));
	  TrueNumInteractions.push_back( d );
	}
      }
    
    }
  }


} //namespace edm
