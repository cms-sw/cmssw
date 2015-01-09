#ifndef _ALCAELECTRONTRACKSREDUCER_H
#define _ALCAELECTRONTRACKSREDUCER_H

// -*- C++ -*-
//
// Package:    AlCaElectronTracksReducer
// Class:      AlCaElectronTracksReducer
// 
/**\class AlCaECALRecHitReducer AlCaECALRecHitReducer.cc Calibration/EcalAlCaRecoProducers/src/AlCaECALRecHitReducer.cc

 Description: This plugin saves tracks and trackExtras that are associated to an electron creating two new track and track extra collections

*/
//
// Original Author:  Shervin Nourbakhsh
//         Created:  Sat Feb 23 10:07:01 CEST 2013
// $Id: AlCaElectronTracksReducer.h,v 1.00 2013/02/23 10:10:34 shervin Exp $
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

// input collections
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"

class AlCaElectronTracksReducer : public edm::EDProducer {
 public:

  explicit AlCaElectronTracksReducer(const edm::ParameterSet&);
  ~AlCaElectronTracksReducer();
  
  virtual void produce(edm::Event &, const edm::EventSetup&);

private:
  // ----------member data ---------------------------
  // input collections
  edm::EDGetTokenT<reco::GsfElectronCollection> electronToken_;
  edm::EDGetTokenT<reco::TrackCollection> generalTracksToken_;
  edm::EDGetTokenT<reco::TrackExtraCollection> generalTracksExtraToken_;
  
  // output collection' names
  std::string alcaTrackCollection_;
  std::string alcaTrackExtraCollection_;

};

#endif
