#ifndef RecoEcal_EgammaClusterProducers_PreshowerClusterProducer_h
#define RecoEcal_EgammaClusterProducers_PreshowerClusterProducer_h

#include <memory>

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/EgammaReco/interface/PreshowerCluster.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "RecoEcal/EgammaClusterAlgos/interface/PreshowerClusterAlgo.h"
#include "CondFormats/ESObjects/interface/ESGain.h"
#include "CondFormats/ESObjects/interface/ESMIPToGeVConstant.h"
#include "CondFormats/ESObjects/interface/ESEEIntercalibConstants.h"
#include "CondFormats/ESObjects/interface/ESMissingEnergyCalibration.h"
#include "CondFormats/ESObjects/interface/ESChannelStatus.h"

class PreshowerClusterProducer : public edm::stream::EDProducer<> {

 public:

  typedef math::XYZPoint Point;

  explicit PreshowerClusterProducer (const edm::ParameterSet& ps);

  ~PreshowerClusterProducer();

  virtual void produce( edm::Event& evt, const edm::EventSetup& es);
  void set(const edm::EventSetup& es);

 private:

  int nEvt_;         // internal counter of events

  //clustering parameters:
  edm::EDGetTokenT<EcalRecHitCollection> preshHitsToken_;         // name of module/plugin/producer producing hits
  edm::EDGetTokenT<reco::SuperClusterCollection> endcapSClusterToken_;   // ditto SuperClusters

  // name out output collections
  std::string preshClusterCollectionX_;  
  std::string preshClusterCollectionY_;  

  int preshNclust_;
  float preshClustECut;
  double etThresh_;

  // association parameters:
  std::string assocSClusterCollection_;    // name of super cluster output collection

  edm::ESHandle<ESGain> esgain_;
  edm::ESHandle<ESMIPToGeVConstant> esMIPToGeV_;
  edm::ESHandle<ESEEIntercalibConstants> esEEInterCalib_;
  edm::ESHandle<ESMissingEnergyCalibration> esMissingECalib_;
  edm::ESHandle<ESChannelStatus> esChannelStatus_;
  double mip_;
  double gamma0_;
  double gamma1_;
  double gamma2_;
  double gamma3_;
  double alpha0_;
  double alpha1_;
  double alpha2_;
  double alpha3_;
  double aEta_[4];
  double bEta_[4];

  PreshowerClusterAlgo * presh_algo; // algorithm doing the real work
   // The set of used DetID's
  //std::set<DetId> used_strips;


};
#endif

