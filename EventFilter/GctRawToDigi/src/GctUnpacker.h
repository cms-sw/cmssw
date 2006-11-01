#ifndef GctUnpacker_h
#define GctUnpacker_h

// -*- C++ -*-
//
// Package:    GctUnpacker
// Class:      GctUnpacker
// 
/**\class GctUnpacker GctUnpacker.cc GctRawToDigi/GctUnpacker/src/GctUnpacker.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jim Brooke
//         Created:  Wed Nov  1 11:57:10 CET 2006
// $Id$
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

//
// class decleration
//

class GctUnpacker : public edm::EDProducer {
   public:
      explicit GctUnpacker(const edm::ParameterSet&);
      ~GctUnpacker();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------
};

#endif
