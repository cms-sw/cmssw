#ifndef MuonIdentification_MuonMerger_h
#define MuonIdentification_MuonMerger_h
//
// Original Author:  Dmytro Kovalskyi
// $Id$
//

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

class MuonMerger : public edm::EDProducer {
 public:
   explicit MuonMerger(const edm::ParameterSet&);
   
   virtual ~MuonMerger();
   
   virtual void produce(edm::Event&, const edm::EventSetup&);

};
#endif
