#include "PhysicsTools/PatAlgos/interface/VertexingHelper.h"
#include <algorithm>

#include <iostream>

#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

pat::helper::VertexingHelper::VertexingHelper(const edm::ParameterSet &iConfig, edm::ConsumesCollector && iC)
{
    if (!iConfig.empty()) {
        enabled_ = true;
        if ( iConfig.existsAs<edm::InputTag>("vertexAssociations") == iConfig.existsAs<edm::InputTag>("vertices")) {
            throw cms::Exception("Configuration") <<
                "VertexingHelper: you must configure either 'vertices' (to produce associations) or 'vertexAssociations' (to read them from disk), " <<
                "you can't specify both, nor you can specify none!\n";
        }

        if (iConfig.existsAs<edm::InputTag>("vertexAssociations")) {
            playback_ = true;
            vertexAssociationsToken_ = iC.consumes<edm::ValueMap<pat::VertexAssociation> >(iConfig.getParameter<edm::InputTag>("vertexAssociations"));
        }
        if (iConfig.existsAs<edm::InputTag>("vertices")) { // vertex have been specified, so run on the fly
            playback_ = false;
            verticesToken_ =  iC.consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertices"));
            // ------ MODE ------------------
            useTracks_ = iConfig.getParameter<bool>("useTracks");
            // ------ CUTS (fully optional) ------------------
        }
        assoSelector_ = reco::modules::make<pat::VertexAssociationSelector>(iConfig);
    } else {
        enabled_ = false;
    }
}

void
pat::helper::VertexingHelper::newEvent(const edm::Event &iEvent) {
    if (playback_) {
        iEvent.getByToken(vertexAssociationsToken_, vertexAssoMap_);
    } else {
        iEvent.getByToken(verticesToken_, vertexHandle_);
    }
}

void
pat::helper::VertexingHelper::newEvent(const edm::Event &iEvent, const edm::EventSetup & iSetup) {
    newEvent(iEvent);
    if (!playback_) iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder", ttBuilder_);
}


pat::VertexAssociation
pat::helper::VertexingHelper::associate(const reco::Candidate &c) const {
    if (playback_) throw cms::Exception("Configuration") << "VertexingHelper: if this module was configured to read associations from the event," <<
                                                            " you must use 'operator()' passing a candidate ref, and not 'associate()' directly!\n";

    reco::VertexCollection::const_iterator vtx, end;
    size_t ivtx;
    reco::TrackBaseRef tk;
    reco::TransientTrack tt;
    if (useTracks_) {
        if (!ttBuilder_.isValid()) throw cms::Exception("Configuration") << "VertexingHelper: If you use 'useTracks', you must call newEvent(iEvent,iSetup)!\n";
        tk = getTrack_(c);
        if (tk.isNull()) return pat::VertexAssociation();
        tt = ttBuilder_->build(*tk);
    }
    for (vtx = vertexHandle_->begin(), end = vertexHandle_->end(), ivtx = 0; vtx != end; ++vtx, ++ivtx) {
        pat::VertexAssociation association(reco::VertexRef(vertexHandle_, ivtx), tk);
        if (useTracks_ == false) {
            association.setDistances(c.vertex(), vtx->position(), vtx->error());
        } else {
            GlobalPoint vtxGP(vtx->x(), vtx->y(), vtx->z()); // need to convert XYZPoint to GlobalPoint
            TrajectoryStateClosestToPoint tscp = tt.trajectoryStateClosestToPoint(vtxGP);
            GlobalPoint          trackPos = tscp.theState().position();
            AlgebraicSymMatrix33 trackErr = tscp.theState().cartesianError().matrix().Sub<AlgebraicSymMatrix33>(0,0);
            association.setDistances(trackPos, vtx->position(), trackErr + vtx->error());
        }
        if (assoSelector_(association)) return association;
    }
    return pat::VertexAssociation();
}

reco::TrackBaseRef pat::helper::VertexingHelper::getTrack_(const reco::Candidate &c) const  {
    const reco::RecoCandidate   *rc  = dynamic_cast<const reco::RecoCandidate *>(&c);
    if (rc  != 0)  { return rc->bestTrackRef(); }
    const reco::PFCandidate *pfc = dynamic_cast<const reco::PFCandidate *>(&c);
    if (pfc != 0) { return reco::TrackBaseRef(pfc->trackRef()); }

    return reco::TrackBaseRef();
}
