#ifndef _ALCAELECTRONTRACKSREDUCER_H
#define _ALCAELECTRONTRACKSREDUCER_H

// -*- C++ -*-
//
// Package:    AlCaElectronTracksReducer
// Class:      AlCaElectronTracksReducer
// 
/**\class AlCaECALRecHitReducer AlCaECALRecHitReducer.cc Calibration/EcalAlCaRecoProducers/src/AlCaECALRecHitReducer.cc

 Description: Example of a producer of AlCa electrons

 Implementation:
     <Notes on implementation>

*/
//
// Original Author:  Shervin Nourbakhsh
//         Created:  Sat Feb 23 10:07:01 CEST 2013
// $Id: AlCaElectronTracksReducer.h,v 1.00 2013/02/23 10:10:34 shervin Exp $
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

class AlCaElectronTracksReducer : public edm::EDProducer {
 public:
  //! ctor
  explicit AlCaElectronTracksReducer(const edm::ParameterSet&);
  ~AlCaElectronTracksReducer();
  
  virtual void produce(edm::Event &, const edm::EventSetup&);
  
 private:
  // ----------member data ---------------------------
  
  edm::InputTag electronLabel_;  
  edm::InputTag generalTracksLabel_;
  std::string alcaTrackCollection_;
  edm::InputTag generalTracksExtraLabel_;
  std::string alcaTrackExtraCollection_;

};

#endif
