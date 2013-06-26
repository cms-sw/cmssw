// -*- C++ -*-
//
// Package:    L1FakeRctProducer
// Class:      FakeGctInputProducer
// 
/**\class FakeGctInputProducer FakeGctInputProducer.h L1Trigger/GlobalCaloTrigger/src/FakeGctInputProducer.h

 \brief EDProducer to fill GCT input buffers for testing

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jim Brooke
//         Created:  Thu Nov 16 00:07:32 CET 2006
// $Id: FakeGctInputProducer.h,v 1.3 2010/01/18 11:53:05 heath Exp $
//
//


#ifndef FAKEGCTINPUTPRODUCER_H
#define FAKEGCTINPUTPRODUCER_H

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//
// class decleration
//

class FakeGctInputProducer : public edm::EDProducer {
   public:
      explicit FakeGctInputProducer(const edm::ParameterSet&);
      ~FakeGctInputProducer();

   private:
      virtual void produce(edm::Event&, const edm::EventSetup&);

      int rgnMode_;
      int iemMode_;
      int niemMode_;

};

#endif
