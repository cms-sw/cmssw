// -*- C++ -*-
//
// Package:    HLTrigger/Muon
// Class:      HLTScoutingTrackProducer
//
/**\class HLTScoutingTrackProducer HLTScoutingTrackProducer.cc HLTScoutingTrackProducer.cc

Description: Producer for Scouting Tracks

*/
//

#include "HLTScoutingTrackProducer.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "TMath.h"

//
// constructors and destructor
//
HLTScoutingTrackProducer::HLTScoutingTrackProducer(const edm::ParameterSet& iConfig):
    OtherTrackCollection_(consumes<reco::TrackCollection>
                     (iConfig.getParameter<edm::InputTag>("OtherTracks")))
{
    //register products
    produces<ScoutingTrackCollection>();
}

HLTScoutingTrackProducer::~HLTScoutingTrackProducer() = default;

// ------------ method called to produce the data  ------------
void HLTScoutingTrackProducer::produce(edm::StreamID sid, edm::Event & iEvent,
                                      edm::EventSetup const & setup) const
{
    using namespace edm;

    std::unique_ptr<ScoutingTrackCollection> outTrack(new ScoutingTrackCollection());
  
    Handle<reco::TrackCollection> OtherTrackCollection;
    
    if(iEvent.getByToken(OtherTrackCollection_, OtherTrackCollection)){
      // Produce tracks in event
      for (auto &trk : *OtherTrackCollection) {
	std::cout << "\n\n\ngot a track "  << trk.pt()  << " " << trk.eta()  << " " << trk.phi()  << "\n\n\n" << std::endl;
	outTrack->emplace_back(trk.pt(), trk.eta(), trk.phi(),trk.chi2(), trk.ndof(),
		               trk.charge(), trk.dxy(), trk.dz(), trk.hitPattern().numberOfValidPixelHits(), 
			       trk.hitPattern().trackerLayersWithMeasurement(), trk.hitPattern().numberOfValidStripHits(), 
			       trk.qoverp(), trk.lambda(), trk.dxyError(),  trk.dzError(),
			       trk.qoverpError(),
			       trk.lambdaError(),
			       trk.phiError(),
			       trk.dsz(),
			       trk.dszError()
			       );
	               
      }
    }
    
    iEvent.put(std::move(outTrack));
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void HLTScoutingTrackProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("OtherTracks", edm::InputTag("hltPixelTracksL3MuonNoVtx"));
    descriptions.add("hltScoutingTrackProducer", desc);
}

// declare this class as a framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTScoutingTrackProducer);
