#ifndef RecoEcal_EgammaClusterProducers_PiZeroDiscriminatorProducer_h
#define RecoEcal_EgammaClusterProducers_PiZeroDiscriminatorProducer_h


#include <memory>

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EgammaReco/interface/PreshowerClusterShape.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"

#include "RecoEcal/EgammaClusterAlgos/interface/EndcapPiZeroDiscriminatorAlgo.h"

#include "TH1.h"
class TFile;


// authors A. Kyriakis, D. Maletic

class PiZeroDiscriminatorProducer : public edm::EDProducer {

 public:

//  typedef math::XYZPoint Point;

  explicit PiZeroDiscriminatorProducer (const edm::ParameterSet& ps);

  ~PiZeroDiscriminatorProducer();

  virtual void produce( edm::Event& evt, const edm::EventSetup& es);

 private:

  int nEvt_;         // internal counter of events

  //clustering parameters:
  edm::EDGetTokenT<reco::PreshowerClusterShapeCollection> pPreshowerShapeClustersXToken_;
  edm::EDGetTokenT<reco::PreshowerClusterShapeCollection> pPreshowerShapeClustersYToken_;

  edm::EDGetTokenT<reco::PhotonCollection> correctedPhotonToken_;
  std::string PhotonPi0DiscriminatorAssociationMap_;

  edm::InputTag barrelRecHitCollection_;
  edm::EDGetTokenT<EcalRecHitCollection> barrelRecHitCollectionToken_;
  edm::InputTag endcapRecHitCollection_;
  edm::EDGetTokenT<EcalRecHitCollection> endcapRecHitCollectionToken_;

  int EScorr_;

  int preshNst_;

  double preshStripECut_;

  double w0_;

  EndcapPiZeroDiscriminatorAlgo * presh_pi0_algo; // algorithm doing the real work
  enum DebugLevel_pi0 { pDEBUG = 0, pINFO = 1, pERROR = 2 };
  DebugLevel_pi0 debugL_pi0;
};
#endif

