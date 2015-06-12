#include "HeavyFlavorAnalysis/Onia2MuMu/interface/OniaVtxReProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/Common/interface/Provenance.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

OniaVtxReProducer::OniaVtxReProducer(const edm::Handle<reco::VertexCollection> &handle, const edm::Event &iEvent) 
{
    const edm::Provenance *prov = handle.provenance();
    if (prov == 0) throw cms::Exception("CorruptData") << "Vertex handle doesn't have provenance.";
//    edm::ParameterSetID psid = prov->psetID();

//    edm::pset::Registry *psregistry = edm::pset::Registry::instance();
    edm::ParameterSet psetFromProvenance = edm::parameterSet(*prov);
//    if (!psregistry->getMapped(psid, psetFromProvenance)) 
//        throw cms::Exception("CorruptData") << "Vertex handle parameter set ID id = " << psid;

    if (edm::moduleName(*prov) != "RecoChargedRefCandidatePrimaryVertexSorter" )
    //if (edm::moduleName(*prov) != "PrimaryVertexProducer")
        throw cms::Exception("Configuration") << "Vertices to re-produce don't come from a PrimaryVertexProducer, but from a " 
              << edm::moduleName(*prov) <<".\n";

    configure(psetFromProvenance); 

    // Now we also dig out the ProcessName used for the reco::Tracks and reco::Vertices
    std::vector<edm::BranchID> parents = prov->parents();
    bool foundTracks = false;
    bool foundBeamSpot = false;
    for (std::vector<edm::BranchID>::const_iterator it = parents.begin(), ed = parents.end(); it != ed; ++it) {
        edm::Provenance parprov = iEvent.getProvenance(*it);
        if (parprov.friendlyClassName() == "recoTracks") {
            tracksTag_ = edm::InputTag(parprov.moduleLabel(), parprov.productInstanceName(), parprov.processName());
            foundTracks = true;
        } else if (parprov.friendlyClassName() == "recoBeamSpot") {
            beamSpotTag_ = edm::InputTag(parprov.moduleLabel(), parprov.productInstanceName(), parprov.processName());
            foundBeamSpot = true;
        }
    }
    if (!foundTracks || !foundBeamSpot) {
        edm::LogWarning("OniaVtxReProducer_MissingParentage") << 
            "Can't find parentage info for vertex collection inputs: " << 
	    (foundTracks ? "" : "tracks ") << (foundBeamSpot ? "" : "beamSpot") << "\n";
    }
}

void
OniaVtxReProducer::configure(const edm::ParameterSet &iConfig) 
{
    config_ = iConfig;
    tracksTag_   = iConfig.getParameter<edm::InputTag>("TrackLabel");
    beamSpotTag_ = iConfig.getParameter<edm::InputTag>("beamSpotLabel");
    algo_.reset(new PrimaryVertexProducerAlgorithm(iConfig)); 
}

std::vector<TransientVertex> 
OniaVtxReProducer::makeVertices(const reco::TrackCollection &tracks, 
                               const reco::BeamSpot &bs, 
                               const edm::EventSetup &iSetup) const 
{
    edm::ESHandle<TransientTrackBuilder> theB;
    iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",theB);

    std::vector<reco::TransientTrack> t_tks; t_tks.reserve(tracks.size());
    for (reco::TrackCollection::const_iterator it = tracks.begin(), ed = tracks.end(); it != ed; ++it) {
        t_tks.push_back((*theB).build(*it));
        t_tks.back().setBeamSpot(bs);
    }

    return algo_->vertices(t_tks, bs,"KalmanVertexFitter");
}
