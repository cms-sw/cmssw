#include "RecoJets/JetPlusTracks/plugins/TrackFromPFProducer.h"
// system include files
#include <memory>
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"

#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

TrackFromPFProducer::TrackFromPFProducer(const edm::ParameterSet& iConfig)
{
 tokenPFCandidates_=consumes<pat::PackedCandidateCollection>(iConfig.getParameter<edm::InputTag>( "PFCandidates" )); 
 tokenPFCandidatesLostTracks_=consumes<pat::PackedCandidateCollection>(iConfig.getParameter<edm::InputTag>( "PFCandidatesLostTracks" ));
 produces<reco::TrackCollection>("tracksFromPF");
}


void TrackFromPFProducer::produce(edm::Event& theEvent, const edm::EventSetup& setup)
{
  //
  // create empty output collections
  //
  auto outputTColl = std::make_unique<reco::TrackCollection>();

    // std::cout<<" TrackFromPFProducer::produce "<<std::endl;
    edm::Handle<pat::PackedCandidateCollection> pfCandidates;
    theEvent.getByToken( tokenPFCandidates_, pfCandidates);

    for(unsigned int i = 0, n = pfCandidates->size(); i < n; ++i) {
        const pat::PackedCandidate &pf = (*pfCandidates)[i];
        if(pf.hasTrackDetails()){
          const reco::Track &mytrack = pf.pseudoTrack();

         // std::cout<<" Track "<<std::isfinite(mytrack.pt())<<" "<<
           //    isnan(mytrack.eta())<<" "<<pf.ptTrk()<<" "<<pf.etaAtVtx()<<" "<<
             // mytrack.found()<<" "<<mytrack.pt()<<" "<<mytrack.eta()<<" "<<mytrack.phi()<<" "<<
               //                 mytrack.vx()<<" "<<mytrack.vy()<<" "<<
                 //               mytrack.vz()<<std::endl;
           if(isnan(mytrack.pt())||isnan(mytrack.eta())||isnan(mytrack.phi())) continue;
           if(!std::isfinite(mytrack.pt()) || !std::isfinite(mytrack.eta()) || !std::isfinite(mytrack.phi())) continue;
           // std::cout<<" Accept "<<std::endl;
           outputTColl->push_back(mytrack);          
          }  
           //else {
           //  if(pf.charge()>0) std::cout<<pf.vx()<<" "<<pf.pt() <<std::endl;
           // }
    }

  //put everything in the event
  theEvent.put(std::move(outputTColl),"tracksFromPF");

}
DEFINE_FWK_MODULE(TrackFromPFProducer);
