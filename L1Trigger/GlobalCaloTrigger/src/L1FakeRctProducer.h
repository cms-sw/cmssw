#ifndef L1FAKERCTPRODUCER_H
#define L1FAKERCTPRODUCER_H


// -*- C++ -*-
//
// Package:    L1FakeRctProducer
// Class:      L1FakeRctProducer
// 
/**\class L1FakeRctProducer L1FakeRctProducer.cc L1Trigger/L1FakeRctProducer/src/L1FakeRctProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jim Brooke
//         Created:  Thu Nov 16 00:07:32 CET 2006
// $Id$
//
//


// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//
// class decleration
//

class L1FakeRctProducer : public edm::EDProducer {
   public:
      explicit L1FakeRctProducer(const edm::ParameterSet&);
      ~L1FakeRctProducer();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      int rgnMode_;
      int iemMode_;
      int niemMode_;

};

#endif
