#ifndef EventFilter_EcalRawToDigi_EcalRawToRecHitProducer_H
#define EventFilter_EcalRawToDigi_EcalRawToRecHitProducer_H
// -*- C++ -*-
//
// Package:    EcalRawToRecHitProducer
// Class:      EcalRawToRecHitProducer
// 
/**\class EcalRawToRecHitProducer EcalRawToRecHitProducer.cc EventFilter/EcalRawToRecHitProducer/src/EcalRawToRecHitProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jean-Roch Vlimant
//         Created:  Sat Oct  6 22:39:47 CEST 2007
// $Id: EcalRawToRecHitProducer.h,v 1.1 2007/12/06 20:01:11 vlimant Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalRecHit/interface/LazyGetter.h"
#include "DataFormats/EcalRecHit/interface/RefGetter.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
//#include "EventFilter/EcalRawToDigi/interface/EcalRegionCablingRecord.h"
//#include "EventFilter/EcalRawToDigi/interface/EcalRegionCabling.h"
//#include "EventFilter/EcalRawToDigi/interface/EcalRawToRecHitLazyUnpacker.h"

#include "EventFilter/EcalRawToDigi/interface/MyWatcher.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitComparison.h"

#include "DataFormats/EcalRawData/interface/EcalListOfFEDS.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TStopwatch.h"

class EcalRawToRecHitProducer : public edm::EDProducer {
   public:
  
  typedef edm::LazyGetter<EcalRecHit> EcalRecHitLazyGetter;
  typedef edm::RefGetter<EcalRecHit> EcalRecHitRefGetter;
  
  explicit EcalRawToRecHitProducer(const edm::ParameterSet&);
  ~EcalRawToRecHitProducer();
  
private:
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  // ----------member data ---------------------------
  edm::InputTag lsourceTag_;
  edm::InputTag sourceTag_;
  edm::InputTag fedListTag_;
  
  bool global_;
  bool splitOutput_;
  std::string EBrechitCollection_;
  std::string EErechitCollection_;
  
};

#endif
