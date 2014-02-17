#ifndef MaskedRctInputDigiProducer_h
#define MaskedRctInputDigiProducer_h

// -*- C++ -*-
//
// Package:    MaskedRctInputDigiProducer
// Class:      MaskedRctInputDigiProducer
// 
/**\class MaskedRctInputDigiProducer MaskedRctInputDigiProducer.cc L1Trigger/MaskedRctInputDigiProducer/src/MaskedRctInputDigiProducer.cc

 Description: Takes collections of ECAL and HCAL digis, masks some towers 
according to a mask in a text file, and creates new collections for use by the
RCT.  

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  pts/65
//         Created:  Fri Nov 23 12:08:31 CET 2007
// $Id: MaskedRctInputDigiProducer.h,v 1.2 2010/01/07 11:10:03 bachtis Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

//
// class declaration
//

class MaskedRctInputDigiProducer : public edm::EDProducer {
public:
  explicit MaskedRctInputDigiProducer(const edm::ParameterSet&);
  ~MaskedRctInputDigiProducer();
  
private:
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
      // ----------member data ---------------------------

  bool useEcal_;
  bool useHcal_;
  edm::InputTag ecalDigisLabel_;
  edm::InputTag hcalDigisLabel_;
  edm::FileInPath maskFile_;
};
#endif
