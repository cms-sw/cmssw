#ifndef RecoEcal_EgammaClusterProducers_PreshowerClusterShapeProducer_h
#define RecoEcal_EgammaClusterProducers_PreshowerClusterShapeProducer_h

// $Author: akyriaki $
// $Id: PreshowerClusterShapeProducer.h,v 1.2 2007/12/07 09:35:52 akyriaki Exp $
// $Date: 2007/12/07 09:35:52 $

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

  edm::InputTag preshHitProducer_;   // name of module/plugin/producer producing hits
  edm::InputTag endcapSClusterProducer_; // likewise for producer of endcap superclusters

//  std::string photonCorrCollectionProducer_;
//  std::string correctedPhotonCollection_;

  std::string PreshowerClusterShapeCollectionX_;
  std::string PreshowerClusterShapeCollectionY_;
  
  EndcapPiZeroDiscriminatorAlgo * presh_pi0_algo; // algorithm doing the real work

  EndcapPiZeroDiscriminatorAlgo::DebugLevel_pi0 debugL_pi0;
};
#endif

