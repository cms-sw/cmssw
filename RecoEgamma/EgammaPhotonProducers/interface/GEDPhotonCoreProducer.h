#ifndef RecoEgamma_EgammaPhotonProducers_GEDPhotonCoreProducer_h
#define RecoEgamma_EgammaPhotonProducers_GEDPhotonCoreProducer_h
/** \class GEDPhotonCoreProducer
 **  
 **
 **  \author Nancy Marinelli, U. of Notre Dame, US
 **
 ***/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/EgammaCandidates/interface/PhotonCore.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
//#include "RecoEgamma/EgammaTools/interface/HoECalculator.h"
//#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
//#include "RecoEgamma/PhotonIdentification/interface/PhotonIsolationCalculator.h"

// GEDPhotonCoreProducer inherits from EDProducer, so it can be a module:
class GEDPhotonCoreProducer : public edm::EDProducer {

 public:

  GEDPhotonCoreProducer (const edm::ParameterSet& ps);
  ~GEDPhotonCoreProducer();

  virtual void produce(edm::Event& evt, const edm::EventSetup& es);

 private:

  void createSingleLegConversions( reco::CaloClusterPtr, const std::vector<reco::TrackRef>&, const std::vector<float>&,  reco::ConversionCollection &oneLegConversions  );

  std::string GEDPhotonCoreCollection_;
  std::string PFConversionCollection_;
  edm::EDGetTokenT<reco::PFCandidateCollection> pfEgammaCandidates_;
  edm::EDGetTokenT<reco::ElectronSeedCollection> pixelSeedProducer_;

  double minSCEt_;
  bool validConversions_;
  edm::ParameterSet conf_;
  bool validPixelSeeds_;

};
#endif
