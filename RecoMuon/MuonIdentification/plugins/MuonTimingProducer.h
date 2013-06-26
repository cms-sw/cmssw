#ifndef MuonTiming_MuonTimingProducer_h
#define MuonTiming_MuonTimingProducer_h 1

// -*- C++ -*-
//
// Package:    MuonTimingProducer
// Class:      MuonTimingProducer
// 
/**\class MuonTimingProducer MuonTimingProducer.h RecoMuon/MuonIdentification/interface/MuonTimingProducer.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Piotr Traczyk, CERN
//         Created:  Mon Mar 16 12:27:22 CET 2009
// $Id: MuonTimingProducer.h,v 1.2 2009/03/27 02:23:58 ptraczyk Exp $
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

#include "DataFormats/MuonReco/interface/MuonTimeExtra.h"
#include "RecoMuon/MuonIdentification/interface/MuonTimingFiller.h"


//
// class decleration
//

class MuonTimingProducer : public edm::EDProducer {
   public:
      explicit MuonTimingProducer(const edm::ParameterSet&);
      ~MuonTimingProducer();

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------
      edm::InputTag m_muonCollection;

      MuonTimingFiller* theTimingFiller_;

};

#endif
