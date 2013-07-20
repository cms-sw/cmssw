#ifndef MuonIdentification_MuonLinksProducerForHLT_h
#define MuonIdentification_MuonLinksProducerForHLT_h

/** \class MuonLinksProducerForHLT
 *
 * Simple producer to make reco::MuonTrackLinks collection 
 * out of the global muons from "muons" collection to restore
 * dropped links used as input for MuonIdProducer.
 *
 *  $Date: 2011/05/03 09:17:49 $
 *  $Revision: 1.2 $
 *  \author R. Bellan - UCSB <riccardo.bellan@cern.ch>
 */


// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class MuonLinksProducerForHLT : public edm::EDProducer {
 public:
   explicit MuonLinksProducerForHLT(const edm::ParameterSet&);
   
   virtual ~MuonLinksProducerForHLT();
   
   virtual void produce(edm::Event&, const edm::EventSetup&);

 private:
   edm::InputTag theLinkCollectionInInput;
   edm::InputTag theInclusiveTrackCollectionInInput;
   double ptMin;
   double pMin;
   double shareHitFraction;
};
#endif
