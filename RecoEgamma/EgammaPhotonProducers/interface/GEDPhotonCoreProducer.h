#ifndef RecoEgamma_EgammaPhotonProducers_GEDPhotonCoreProducer_h
#define RecoEgamma_EgammaPhotonProducers_GEDPhotonCoreProducer_h
/** \class GEDPhotonCoreProducer
 **  
 **
 **  $Id: GEDPhotonCoreProducer.h,v 1.1 2013/05/07 12:34:14 nancy Exp $ 
 **  $Date: 2013/05/07 12:34:14 $ 
 **  $Revision: 1.1 $
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

  void createSingleLegConversions( reco::CaloClusterPtr, std::vector<reco::TrackRef>, std::vector<float>,  reco::ConversionCollection &oneLegConversions  );

  std::string GEDPhotonCoreCollection_;
  std::string PFConversionCollection_;
  edm::InputTag pfEgammaCandidates_;

  double minSCEt_;
  bool validConversions_;
  std::string pixelSeedProducer_;
  edm::ParameterSet conf_;
  bool validPixelSeeds_;

};
#endif
