#ifndef _ALCAECALRECHITREDUCER_H
#define _ALCAECALRECHITREDUCER_H

// -*- C++ -*-
//
// Package:    AlCaECALRecHitReducer
// Class:      AlCaECALRecHitReducer
// 
/**\class AlCaECALRecHitReducer AlCaECALRecHitReducer.cc Calibration/EcalAlCaRecoProducers/src/AlCaECALRecHitReducer.cc

 Description: Example of a producer of AlCa electrons

 Implementation:
     <Notes on implementation>

*/
//
// Original Author:  Lorenzo AGOSTINO
//         Created:  Mon Jul 17 18:07:01 CEST 2006
// $Id: AlCaECALRecHitReducer.h,v 1.13 2010/02/11 00:10:34 wmtan Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
//!
//! class declaration
//!

class AlCaECALRecHitReducer : public edm::EDProducer {
 public:
  //! ctor
  explicit AlCaECALRecHitReducer(const edm::ParameterSet&);
  ~AlCaECALRecHitReducer();
  
  
  //! producer
  virtual void produce(edm::Event &, const edm::EventSetup&);
  
 private:
  // ----------member data ---------------------------
  
  
  
  edm::EDGetTokenT<EcalRecHitCollection> ebRecHitsToken_;
  edm::EDGetTokenT<EcalRecHitCollection> eeRecHitsToken_;
  edm::EDGetTokenT<EcalRecHitCollection> esRecHitsToken_;
  edm::EDGetTokenT<reco::GsfElectronCollection> electronToken_;
  std::vector< edm::EDGetTokenT<edm::View < reco::RecoCandidate> > > eleViewTokens_;

  edm::EDGetTokenT<reco::PhotonCollection> photonToken_;
  edm::EDGetTokenT<reco::SuperClusterCollection> EESuperClusterToken_;
  std::string alcaBarrelHitsCollection_;
  std::string alcaEndcapHitsCollection_;
  std::string alcaPreshowerHitsCollection_;
  int etaSize_;
  int phiSize_;
  //float weight_;
  //  int esNstrips_;
  //int esNcolumns_;

  //  bool selectByEleNum_;
  //  int minEleNumber_;
  //  double minElePt_;
  double minEta_highEtaSC_;
  std::string alcaCaloClusterCollection_;

  void AddMiniRecHitCollection(const reco::SuperCluster& sc,
			       std::set<DetId>& reducedRecHitMap,
			       const CaloTopology *caloTopology
			       );


};

#endif
