#ifndef RecoEgamma_EgammaTools_ElectronEnergyCalibrator_h
#define RecoEgamma_EgammaTools_ElectronEnergyCalibrator_h

//author: Alan Smithee
//description: 
//  this class allows the residual scale and smearing to be applied to electrons
//  the scale and smearing is on the ecal part of the energy
//  hence the E/p combination needs to be re-don, hence the E/p Combination Tools
//  it re-applies the regression with the new corrected ecal energy
//  returns a vector of calibrated energies and correction data, indexed by EGEnergySysIndex
//  a port of EgammaAnalysis/ElectronTools/ElectronEnergyCalibratorRun2

#include "FWCore/Utilities/interface/StreamID.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "RecoEgamma/EgammaTools/interface/EnergyScaleCorrection.h"
#include "RecoEgamma/EgammaTools/interface/EpCombinationTool.h"
#include "RecoEgamma/EgammaTools/interface/EGEnergySysIndex.h"

#include <TRandom.h>

#include <vector>

class ElectronEnergyCalibrator
{
public:
  enum class EventType{
    DATA,
    MC,
  };

  ElectronEnergyCalibrator() {}
  ElectronEnergyCalibrator(const EpCombinationTool &combinator, const std::string& correctionFile );
  ~ElectronEnergyCalibrator() {}
  
  /// Initialize with a random number generator (if not done, it will use the CMSSW service)
  /// Caller code owns the TRandom.
  void initPrivateRng(TRandom *rnd) ;

  //set the minimum et to apply the correction to
  void setMinEt(float val){minEt_=val;}
  //sets whether to use the smeared ecal energy in the combination
  //note, if this is true and the E/p combination is not trained using this smeared value, this is a bug
  //the E/p combination must get the ecalEnergyErr used in its training
  void setUseSmearCorrEcalEnergyErrInComb(bool val){useSmearCorrEcalEnergyErrInComb_=val;}
  /// Correct this electron.
  /// StreamID is needed when used with CMSSW Random Number Generator
  std::array<float,EGEnergySysIndex::kNrSysErrs> 
  calibrate(reco::GsfElectron &ele, const unsigned int runNumber, 
	    const EcalRecHitCollection* recHits, edm::StreamID const & id, 
	    const EventType eventType) const ;

  std::array<float,EGEnergySysIndex::kNrSysErrs>
  calibrate(reco::GsfElectron &ele, const unsigned int runNumber, 
	    const EcalRecHitCollection* recHits, const float smearNrSigma, 
	    const EventType eventType) const ;

private:
  void setEnergyAndSystVarations(const float scale,const float smearNrSigma,const float et,
				 const EnergyScaleCorrection::ScaleCorrection& scaleCorr,
				 const EnergyScaleCorrection::SmearCorrection& smearCorr,
				 reco::GsfElectron& ele,
				 std::array<float,EGEnergySysIndex::kNrSysErrs>& energyData)const;
    
  void setEcalEnergy(reco::GsfElectron& ele,const float scale,const float smear)const;
  std::pair<float,float> calCombinedMom(reco::GsfElectron& ele,const float scale,const float smear)const;
  
  /// Return a number distributed as a unit gaussian, drawn from the private RNG if initPrivateRng was called,
  /// or from the CMSSW RandomNumberGenerator service
  /// If synchronization is set to true, it returns a fixed number (1.0)
  double gauss(edm::StreamID const& id) const ;
  
  // whatever data will be needed
  EnergyScaleCorrection correctionRetriever_;
  const EpCombinationTool *epCombinationTool_; //this is not owned
  TRandom *rng_; //this is not owned
  float minEt_;
  bool useSmearCorrEcalEnergyErrInComb_;

  //default values to access if no correction available
  static const EnergyScaleCorrection::ScaleCorrection defaultScaleCorr_;
  static const EnergyScaleCorrection::SmearCorrection defaultSmearCorr_;
  

};

#endif
