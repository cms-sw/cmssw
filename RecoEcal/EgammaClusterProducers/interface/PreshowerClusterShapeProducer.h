#ifndef RecoEcal_EgammaClusterProducers_PreshowerClusterShapeProducer_h
#define RecoEcal_EgammaClusterProducers_PreshowerClusterShapeProducer_h

// $Author: akyriaki $
// $Id: PreshowerClusterShapeProducer.h,v 1.1 2007/10/18 13:26:49 akyriaki Exp $
// $Date: 2007/10/18 13:26:49 $

#include <memory>

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

#include "DataFormats/EgammaReco/interface/PreshowerClusterShape.h"
#include "RecoEcal/EgammaClusterAlgos/interface/EndcapPiZeroDiscriminatorAlgo.h"

// authors A. Kyriakis, D. Maletic

class PreshowerClusterShapeProducer : public edm::EDProducer {

 public:

  typedef math::XYZPoint Point;

  explicit PreshowerClusterShapeProducer (const edm::ParameterSet& ps);

  ~PreshowerClusterShapeProducer();

  virtual void produce( edm::Event& evt, const edm::EventSetup& es);

 private:

  int nEvt_;         // internal counter of events

  //clustering parameters:

  std::string preshHitProducer_;   // name of module/plugin/producer producing hits
  std::string preshHitCollection_; // secondary name given to collection of hits by hitProducer

//  std::string photonCorrCollectionProducer_;
//  std::string correctedPhotonCollection_;

  std::string endcapSClusterCollection_;
  std::string endcapSClusterProducer_;

  std::string PreshowerClusterShapeCollectionX_;
  std::string PreshowerClusterShapeCollectionY_;
  
  EndcapPiZeroDiscriminatorAlgo * presh_pi0_algo; // algorithm doing the real work

  EndcapPiZeroDiscriminatorAlgo::DebugLevel_pi0 debugL_pi0;
};
#endif

