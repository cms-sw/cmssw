#ifndef PhysicsTools_PatAlgos_PATMuonCleaner_h
#define PhysicsTools_PatAlgos_PATMuonCleaner_h
//
// $Id: PATMuonCleaner.h,v 1.1 2008/01/15 13:30:07 lowette Exp $
//

/**
  \class    PATMuonCleaner PATMuonCleaner.h "PhysicsTools/PatAlgos/interface/PATMuonCleaner.h"
  \brief    Produces a clean list of muons, and associated back-references to the original muon collection

   The PATMuonCleaner produces a list of clean muons with associated back-references to the original muon collection.
   At the moment it really does <b>no cleaning at all</b>, but this will be implemented later.
 
  \author   Steven Lowette, Roger Wolf, 
  \version  $Id: PATMuonCleaner.h,v 1.1 2008/01/15 13:30:07 lowette Exp $
*/


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include <string>

namespace pat {


  class ObjectResolutionCalc;
  class LeptonLRCalc;


  class PATMuonCleaner : public edm::EDProducer {

    public:

      explicit PATMuonCleaner(const edm::ParameterSet & iConfig);
      ~PATMuonCleaner();

      virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);

    private:
      // configurables
      edm::InputTag muonSrc_;

  };


}

#endif
