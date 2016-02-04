#ifndef RecoEcal_EgammaClusterProducers_PiZeroDiscriminatorProducer_h
#define RecoEcal_EgammaClusterProducers_PiZeroDiscriminatorProducer_h

// $Author: argiro $
// $Id: PiZeroDiscriminatorProducer.h,v 1.4 2011/07/19 10:09:25 argiro Exp $
// $Date: 2011/07/19 10:09:25 $

#include <memory>

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

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
  std::string preshClusterShapeCollectionX_;  // secondary name to be given to collection of cluster produced in this module
  std::string preshClusterShapeCollectionY_;
  std::string preshClusterShapeProducer_;

  std::string photonCorrCollectionProducer_;
  std::string correctedPhotonCollection_;
  std::string PhotonPi0DiscriminatorAssociationMap_;

  edm::InputTag barrelRecHitCollection_;
  edm::InputTag endcapRecHitCollection_;

  int EScorr_; 

  int preshNst_;  
  
  double preshStripECut_;
  
  double w0_; 

  EndcapPiZeroDiscriminatorAlgo * presh_pi0_algo; // algorithm doing the real work
  enum DebugLevel_pi0 { pDEBUG = 0, pINFO = 1, pERROR = 2 };
  DebugLevel_pi0 debugL_pi0;
};
#endif

