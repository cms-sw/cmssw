#ifndef EventFilter_EcalRawToDigi_EcalRawToRecHitFacility_H
#define EventFilter_EcalRawToDigi_EcalRawToRecHitFacility_H

// -*- C++ -*-
//
// Package:    EcalRawToRecHitFacility
// Class:      EcalRawToRecHitFacility
// 
/**\class EcalRawToRecHitFacility EcalRawToRecHitFacility.cc EventFilter/EcalRawToDigi/src/EcalRawToRecHitFacility.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jean-Roch Vlimant
//         Created:  Sat Oct  6 02:26:08 CEST 2007
// $Id: EcalRawToRecHitFacility.h,v 1.1 2007/12/06 20:01:11 vlimant Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalRecHit/interface/LazyGetter.h"
#include "DataFormats/Common/interface/RefGetter.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "EventFilter/EcalRawToDigi/interface/EcalRawToRecHitLazyUnpacker.h"
#include "EventFilter/EcalRawToDigi/interface/EcalRegionCabling.h"
#include "EventFilter/EcalRawToDigi/interface/EcalRegionCablingRecord.h"
#include "EventFilter/EcalRawToDigi/interface/EcalUnpackerWorker.h"
#include "EventFilter/EcalRawToDigi/interface/EcalUnpackerWorkerRecord.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "EventFilter/EcalRawToDigi/interface/MyWatcher.h"


class EcalRawToRecHitFacility : public edm::EDProducer {
public:
  explicit EcalRawToRecHitFacility(const edm::ParameterSet&);
  ~EcalRawToRecHitFacility();
  
  typedef edm::LazyGetter<EcalRecHit> EcalRecHitLazyGetter;
  typedef edm::RefGetter<EcalRecHit> EcalRecHitRefGetter;
  
private:
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  //in
  edm::InputTag sourceTag_;
  
  //tools
  std::string workerName_;
  
  //out
  bool global_;
};
#endif
