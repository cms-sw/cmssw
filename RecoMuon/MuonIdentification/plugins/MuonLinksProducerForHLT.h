#ifndef MuonIdentification_MuonLinksProducerForHLT_h
#define MuonIdentification_MuonLinksProducerForHLT_h

/** \class MuonLinksProducerForHLT
 *
 * Simple producer to make reco::MuonTrackLinks collection 
 * out of the global muons from "muons" collection to restore
 * dropped links used as input for MuonIdProducer.
 *
 *  \author R. Bellan - UCSB <riccardo.bellan@cern.ch>
 */


// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
//#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonTrackLinks.h"

class MuonLinksProducerForHLT : public edm::global::EDProducer<> {
 public:
   explicit MuonLinksProducerForHLT(const edm::ParameterSet&);
   
   virtual ~MuonLinksProducerForHLT();
   
   virtual void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

 private:
   edm::InputTag theLinkCollectionInInput;
   edm::InputTag theInclusiveTrackCollectionInInput;
   edm::EDGetTokenT<reco::MuonTrackLinksCollection> linkToken_;
   edm::EDGetTokenT<reco::TrackCollection> trackToken_;
   double ptMin;
   double pMin;
   double shareHitFraction;
};
#endif
