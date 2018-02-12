#ifndef ElectronEnergyCalibratorRun2_h
#define ElectronEnergyCalibratorRun2_h

#include <TRandom.h>
#include "EgammaAnalysis/ElectronTools/interface/EnergyScaleCorrection_class.h"
#include "EgammaAnalysis/ElectronTools/interface/EpCombinationToolSemi.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include <vector>

class ElectronEnergyCalibratorRun2
{
 public:
  // dummy constructor for persistence
  ElectronEnergyCalibratorRun2() {}
  
  // further configuration will be added when we will learn how it will work
  ElectronEnergyCalibratorRun2(EpCombinationToolSemi &combinator, bool isMC, bool synchronization, std::string);
  ~ElectronEnergyCalibratorRun2() ;
  
  /// Initialize with a random number generator (if not done, it will use the CMSSW service)
  /// Caller code owns the TRandom.
  void initPrivateRng(TRandom *rnd) ;
  
  /// Correct this electron.
  /// StreamID is needed when used with CMSSW Random Number Generator
  std::vector<float> calibrate(reco::GsfElectron &electron, unsigned int runNumber, 
			       const EcalRecHitCollection* recHits, edm::StreamID const & id = edm::StreamID::invalidStreamID(), int eventIsMC = -1) const ;
  
 protected:
  // whatever data will be needed
  EpCombinationToolSemi *epCombinationTool_;
  bool isMC_;
  bool synchronization_;
  TRandom *rng_;
  
  /// Return a number distributed as a unit gaussian, drawn from the private RNG if initPrivateRng was called,
  /// or from the CMSSW RandomNumberGenerator service
  /// If synchronization is set to true, it returns a fixed number (1.0)
  double gauss(edm::StreamID const& id) const ;
  EnergyScaleCorrection_class correctionRetriever_;
};

#endif
