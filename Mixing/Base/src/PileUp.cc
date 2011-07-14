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

} //namespace edm
