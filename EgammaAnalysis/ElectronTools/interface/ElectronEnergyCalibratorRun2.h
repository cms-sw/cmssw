#ifndef ElectronEnergyCalibratorRun2_h
#define ElectronEnergyCalibratorRun2_h

#include <TRandom.h>
#include "EgammaAnalysis/ElectronTools/interface/EnergyScaleCorrection_class.h"
#include "EgammaAnalysis/ElectronTools/interface/SimpleElectron.h"
#include "EgammaAnalysis/ElectronTools/interface/EpCombinationTool.h"
#include "FWCore/Utilities/interface/StreamID.h"


#include <vector>

class ElectronEnergyCalibratorRun2 {
 public:
  // dummy constructor for persistence
  ElectronEnergyCalibratorRun2() {}
  
  // further configuration will be added when we will learn how it will work
  ElectronEnergyCalibratorRun2(EpCombinationTool &combinator, bool isMC, bool synchronization, std::string); 
  ~ElectronEnergyCalibratorRun2() ;
  
  /// Initialize with a random number generator (if not done, it will use the CMSSW service)
  /// Caller code owns the TRandom.
  void initPrivateRng(TRandom *rnd) ;
  
  /// Correct this electron. 
  /// StreamID is needed when used with CMSSW Random Number Generator
  void calibrate(SimpleElectron &electron, edm::StreamID const & id = edm::StreamID::invalidStreamID()) const ;
  
  /// Correct this electron. 
  /// StreamID is needed when used with CMSSW Random Number Generator
  void calibrate(reco::GsfElectron &electron, unsigned int runNumber, edm::StreamID const & id = edm::StreamID::invalidStreamID()) const ;
  
 protected:    
  // whatever data will be needed
  EpCombinationTool *epCombinationTool_;
  bool isMC_, synchronization_;
  TRandom *rng_;
  
  /// Return a number distributed as a unit gaussian, drawn from the private RNG if initPrivateRng was called, 
  /// or from the CMSSW RandomNumberGenerator service
  /// If synchronization is set to true, it returns a fixed number (1.0)
  double gauss(edm::StreamID const& id) const ;
  EnergyScaleCorrection_class _correctionRetriever;
};

#endif
