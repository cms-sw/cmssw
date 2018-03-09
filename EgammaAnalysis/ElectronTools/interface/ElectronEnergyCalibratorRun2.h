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
  void setMinEt(float val){minEt_=val;}
  /// Correct this electron.
  /// StreamID is needed when used with CMSSW Random Number Generator
  std::vector<float> calibrate(reco::GsfElectron &ele, const unsigned int runNumber, 
			       const EcalRecHitCollection* recHits, edm::StreamID const & id = edm::StreamID::invalidStreamID(), const int eventIsMC = -1) const ;
  std::vector<float> calibrate(reco::GsfElectron &ele, const unsigned int runNumber, 
			       const EcalRecHitCollection* recHits, const float smearNrSigma, const int eventIsMC = -1) const ;

private:
  void setEcalEnergy(reco::GsfElectron& ele,const float scale,const float smear)const;
  std::pair<float,float> calCombinedMom(reco::GsfElectron& ele,const float scale,const float smear)const;
  

 protected:
  // whatever data will be needed
  EpCombinationToolSemi *epCombinationTool_; //this is not owned
  bool isMC_;
  bool synchronization_;
  TRandom *rng_; //this is not owned
  float minEt_;
  
  /// Return a number distributed as a unit gaussian, drawn from the private RNG if initPrivateRng was called,
  /// or from the CMSSW RandomNumberGenerator service
  /// If synchronization is set to true, it returns a fixed number (1.0)
  double gauss(edm::StreamID const& id) const ;
  EnergyScaleCorrection_class correctionRetriever_;
};

#endif
