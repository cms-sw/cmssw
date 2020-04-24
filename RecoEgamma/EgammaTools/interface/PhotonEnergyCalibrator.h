#ifndef RecoEgamma_EgammaTools_PhotonEnergyCalibrator_h
#define RecoEgamma_EgammaTools_PhotonEnergyCalibrator_h

//author: Alan Smithee
//description: 
//  this class allows the residual scale and smearing to be applied to photons
//  returns a vector of calibrated energies and correction data, indexed by EGEnergySysIndex
//  a port of EgammaAnalysis/ElectronTools/ElectronEnergyCalibratorRun2

#include "FWCore/Utilities/interface/StreamID.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "RecoEgamma/EgammaTools/interface/EnergyScaleCorrection.h"
#include "RecoEgamma/EgammaTools/interface/EGEnergySysIndex.h"

#include <TRandom.h>

#include <vector>

class PhotonEnergyCalibrator
{
 public: 
  enum class EventType{
    DATA,
    MC,
  };

  PhotonEnergyCalibrator() {}
  PhotonEnergyCalibrator(const std::string& correctionFile);  
  ~PhotonEnergyCalibrator(){}
  
  /// Initialize with a random number generator (if not done, it will use the CMSSW service)
  /// Caller code owns the TRandom.
  void initPrivateRng(TRandom *rnd) ;

  //set the minimum et to apply the correction to
  void setMinEt(float val){minEt_=val;}
  
  /// Correct this photon.
  /// StreamID is needed when used with CMSSW Random Number Generator
  std::array<float,EGEnergySysIndex::kNrSysErrs> 
  calibrate(reco::Photon &photon, const unsigned int runNumber, 
	    const EcalRecHitCollection* recHits,  edm::StreamID const & id, const EventType eventType) const ;

  std::array<float,EGEnergySysIndex::kNrSysErrs> 
  calibrate(reco::Photon &photon, const unsigned int runNumber, 
	    const EcalRecHitCollection* recHits, const float smearNrSigma, 
	    const EventType eventType) const ;
  
private:
  void setEnergyAndSystVarations(const float scale,const float smearNrSigma,const float et,
				 const EnergyScaleCorrection::ScaleCorrection& scaleCorr,
				 const EnergyScaleCorrection::SmearCorrection& smearCorr,
				 reco::Photon& photon,
				 std::array<float,EGEnergySysIndex::kNrSysErrs>& energyData)const;

  /// Return a number distributed as a unit gaussian, drawn from the private RNG if initPrivateRng was called,
  /// or from the CMSSW RandomNumberGenerator service
  /// If synchronization is set to true, it returns a fixed number (1.0)
  double gauss(edm::StreamID const& id) const ;

  // whatever data will be needed
  EnergyScaleCorrection correctionRetriever_;
  TRandom *rng_; //this is not owned
  float minEt_;

  //default values to access if no correction availible
  static const EnergyScaleCorrection::ScaleCorrection defaultScaleCorr_;
  static const EnergyScaleCorrection::SmearCorrection defaultSmearCorr_;
  
};

#endif
