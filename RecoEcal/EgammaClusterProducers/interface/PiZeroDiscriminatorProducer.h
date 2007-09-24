#ifndef RecoEcal_EgammaClusterProducers_PiZeroDiscriminatorProducer_h
#define RecoEcal_EgammaClusterProducers_PiZeroDiscriminatorProducer_h

// $Author: akyriaki $
// $Id: PiZeroDiscriminatorProducer.h,v 1.3 2007/06/25 09:16:03 akyriaki Exp $
// $Date: 2007/06/25 09:16:03 $

#include <memory>

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

#include "RecoEcal/EgammaClusterAlgos/interface/EndcapPiZeroDiscriminatorAlgo.h"
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
#include "RecoEcal/EgammaCoreTools/interface/ClusterShapeAlgo.h"


#include "TH1.h"
class TFile;


// authors A. Kyriakis, D. Maletic

class PiZeroDiscriminatorProducer : public edm::EDProducer {

 public:

  typedef math::XYZPoint Point;

  explicit PiZeroDiscriminatorProducer (const edm::ParameterSet& ps);

  ~PiZeroDiscriminatorProducer();

  virtual void produce( edm::Event& evt, const edm::EventSetup& es);

 private:

  int nEvt_;         // internal counter of events

  //clustering parameters:
  std::string endcapHitProducer_;
  std::string endcapHitCollection_;
  std::string barrelHitProducer_;  
  std::string barrelHitCollection_;  

  std::string preshHitProducer_;   // name of module/plugin/producer producing hits
  std::string preshHitCollection_; // secondary name given to collection of hits by hitProducer

  std::string photonCorrCollectionProducer_;
  std::string correctedPhotonCollection_;
  std::string PhotonPi0DiscriminatorAssociationMap_;

  EndcapPiZeroDiscriminatorAlgo * presh_pi0_algo; // algorithm doing the real work
  PositionCalc posCalculator_; // position calculation algorithm
  ClusterShapeAlgo shapeAlgo_; // cluster shape algorithm

  EndcapPiZeroDiscriminatorAlgo::DebugLevel_pi0 debugL_pi0;
};
#endif

