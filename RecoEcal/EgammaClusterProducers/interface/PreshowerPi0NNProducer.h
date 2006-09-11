#ifndef RecoEcal_EgammaClusterProducers_PreshowerPi0NNProducer_h
#define RecoEcal_EgammaClusterProducers_PreshowerPi0NNProducer_h

// $Author: Aristoteles Kyriakis$
// $Id$
// $Date$

#include <memory>

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

#include "RecoEcal/EgammaClusterAlgos/interface/PreshowerPi0NNAlgo.h"

#include "TH1.h"
class TFile;


class PreshowerPi0NNProducer : public edm::EDProducer {

 public:

  typedef math::XYZPoint Point;

  explicit PreshowerPi0NNProducer (const edm::ParameterSet& ps);

  ~PreshowerPi0NNProducer();

  virtual void produce( edm::Event& evt, const edm::EventSetup& es);

 private:

  int nEvt_;         // internal counter of events

  //clustering parameters:
  std::string endcapHitProducer_;
  std::string endcapHitCollection_;

  std::string preshHitProducer_;   // name of module/plugin/producer producing hits
  std::string preshHitCollection_; // secondary name given to collection of hits by hitProducer
  std::string endcapSClusterCollection_;
  std::string endcapSClusterProducer_;

  bool clustershape_logweighted;
  float clustershape_x0;
  float clustershape_t0;
  float clustershape_w0;

  std::string new_ClusterPi0DiscriminatorCollection_;

  PreshowerPi0NNAlgo * presh_pi0_algo; // algorithm doing the real work
 
  PreshowerPi0NNAlgo::DebugLevel_pi0 debugL_pi0;
};
#endif

