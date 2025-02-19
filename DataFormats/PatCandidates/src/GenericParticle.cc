//
// $Id: GenericParticle.cc,v 1.4 2009/03/25 22:33:29 hegner Exp $
//

#include "DataFormats/PatCandidates/interface/GenericParticle.h"


using pat::GenericParticle;


/// default constructor
GenericParticle::GenericParticle() :
    PATObject<reco::RecoCandidate>()
{
}


/// constructor from Candidate
GenericParticle::GenericParticle(const Candidate & cand) :
    PATObject<reco::RecoCandidate>()
{
    fillInFrom(cand);
}


/// constructor from ref to RecoCandidate
GenericParticle::GenericParticle(const edm::RefToBase<Candidate> & cand) :
    PATObject<reco::RecoCandidate>()
{
    fillInFrom(*cand);
    refToOrig_ = edm::Ptr<reco::Candidate>(cand.id(), cand.get(), cand.key()); // correct RefToBase=>Ptr conversion
}

/// constructor from ref to RecoCandidate
GenericParticle::GenericParticle(const edm::Ptr<Candidate> & cand) :
    PATObject<reco::RecoCandidate>()
{
    fillInFrom(*cand);
    refToOrig_ = cand;
}



/// destructor
GenericParticle::~GenericParticle() {
}

// ====== SETTERS =====
/// sets master track reference (or even embed it into the object)
void GenericParticle::setTrack(const reco::TrackRef &ref, bool embed) {
    trackRef_ = ref;
    if (embed) embedTrack(); else track_.clear();
}

/// sets multiple track references (or even embed the tracks into the object - whatch out for disk size issues!)
void GenericParticle::setTracks(const reco::TrackRefVector &refs, bool embed) {
    trackRefs_ = refs;
    if (embed) embedTracks(); else tracks_.clear();
}

/// sets stand-alone muon track reference (or even embed it into the object)
void GenericParticle::setStandAloneMuon(const reco::TrackRef &ref, bool embed) {
    standaloneTrackRef_ = ref;
    if (embed) embedStandalone(); else standaloneTrack_.clear();
}

/// sets combined muon track reference (or even embed it into the object)
void GenericParticle::setCombinedMuon(const reco::TrackRef &ref, bool embed) {
    combinedTrackRef_ = ref;
    if (embed) embedCombined(); else combinedTrack_.clear();
}

/// sets gsf track reference (or even embed it into the object)
void GenericParticle::setGsfTrack(const reco::GsfTrackRef &ref, bool embed) {
    gsfTrackRef_ = ref;
    if (embed) embedGsfTrack(); else gsfTrack_.clear();
}

/// sets supercluster reference (or even embed it into the object)
void GenericParticle::setSuperCluster(const reco::SuperClusterRef &ref, bool embed) {
    superClusterRef_ = ref;
    if (embed) embedSuperCluster(); else superCluster_.clear();
}

/// sets calotower reference (or even embed it into the object)
void GenericParticle::setCaloTower(const CaloTowerRef &ref, bool embed) {
    caloTowerRef_ = ref;
    if (embed) { 
        embedCaloTower(); 
    } else if (!caloTower_.empty()) { 
        CaloTowerCollection().swap(caloTower_); 
    }
}


// ========== EMBEDDER METHODS
/// embeds the master track instead of keeping a reference to it      
void GenericParticle::embedTrack() { 
    track_.clear();
    if (trackRef_.isNonnull()) track_.push_back(*trackRef_); // import
    trackRef_ = reco::TrackRef(); // clear, to save space (zeroes compress better)
}
/// embeds the other tracks instead of keeping references
void GenericParticle::embedTracks() { 
    tracks_.clear();
    tracks_.reserve(trackRefs_.size());
    for (reco::TrackRefVector::const_iterator it = trackRefs_.begin(); it != trackRefs_.end(); ++it) {
        if (it->isNonnull()) tracks_.push_back(**it); // embed track
    }
    trackRefs_ = reco::TrackRefVector(); // clear, to save space
}
/// embeds the stand-alone track instead of keeping a reference to it      
void GenericParticle::embedStandalone() { 
    standaloneTrack_.clear();
    if (standaloneTrackRef_.isNonnull()) standaloneTrack_.push_back(*standaloneTrackRef_); // import
    standaloneTrackRef_ = reco::TrackRef(); // clear, to save space (zeroes compress better)
}
/// embeds the combined track instead of keeping a reference to it      
void GenericParticle::embedCombined() { 
    combinedTrack_.clear();
    if (combinedTrackRef_.isNonnull()) combinedTrack_.push_back(*combinedTrackRef_); // import
    combinedTrackRef_ = reco::TrackRef(); // clear, to save space (zeroes compress better)
}
/// embeds the gsf track instead of keeping a reference to it      
void GenericParticle::embedGsfTrack() { 
    gsfTrack_.clear();
    if (gsfTrackRef_.isNonnull()) gsfTrack_.push_back(*gsfTrackRef_); // import
    gsfTrackRef_ = reco::GsfTrackRef(); // clear, to save space (zeroes compress better)
}

/// embeds the supercluster instead of keeping a reference to it      
void GenericParticle::embedSuperCluster() { 
    superCluster_.clear();
    if (superClusterRef_.isNonnull()) superCluster_.push_back(*superClusterRef_); // import
    superClusterRef_ = reco::SuperClusterRef(); // clear, to save space (zeroes compress better)
}
/// embeds the calotower instead of keeping a reference to it      
void GenericParticle::embedCaloTower() { 
    if (!caloTower_.empty()) CaloTowerCollection().swap(caloTower_); 
    if (caloTowerRef_.isNonnull()) caloTower_.push_back(*caloTowerRef_); // import
    caloTowerRef_ = CaloTowerRef(); // clear, to save space (zeroes compress better)
}


void GenericParticle::fillInFrom(const reco::Candidate &cand) {
    // first, kinematics & status
    setCharge(cand.charge());
    setP4(cand.polarP4());
    setVertex(cand.vertex());
    setPdgId(cand.pdgId());
    setStatus(cand.status());
    // then RECO part, if available
    const reco::RecoCandidate *rc = dynamic_cast<const reco::RecoCandidate *>(&cand);
    if (rc != 0) {
        setTrack(rc->track());    
        setGsfTrack(rc->gsfTrack());    
        setStandAloneMuon(rc->standAloneMuon());    
        setCombinedMuon(rc->combinedMuon());
        setSuperCluster(rc->superCluster());    
        setCaloTower(rc->caloTower());    
        size_t ntracks = rc->numberOfTracks();
        if (ntracks > 0) {
            reco::TrackRefVector tracks;
            for (size_t i = 0; i < ntracks; ++i) {
                tracks.push_back(rc->track(i));
            }
            setTracks(tracks);
        }
    }
}

bool GenericParticle::overlap( const reco::Candidate &cand ) const {
    const reco::RecoCandidate *rc = dynamic_cast<const reco::RecoCandidate *>(&cand);
    if (rc != 0) {
        if (rc->track().isNonnull()          && (track() == rc->track())) return true;
        if (rc->gsfTrack().isNonnull()       && (gsfTrack() == rc->gsfTrack())) return true;
        if (rc->standAloneMuon().isNonnull() && (standAloneMuon() == rc->standAloneMuon())) return true;
        if (rc->combinedMuon().isNonnull()   && (combinedMuon() == rc->combinedMuon())) return true;
        if (rc->superCluster().isNonnull()   && (superCluster() == rc->superCluster())) return true;
        if (rc->caloTower().isNonnull()      && (caloTower() == rc->caloTower())) return true;
   }
    const GenericParticle *rc2 = dynamic_cast<const GenericParticle *>(&cand);
    if (rc2 != 0) {
        if (rc2->track().isNonnull()          && (track() == rc2->track())) return true;
        if (rc2->gsfTrack().isNonnull()       && (gsfTrack() == rc2->gsfTrack())) return true;
        if (rc2->standAloneMuon().isNonnull() && (standAloneMuon() == rc2->standAloneMuon())) return true;
        if (rc2->combinedMuon().isNonnull()   && (combinedMuon() == rc2->combinedMuon())) return true;
        if (rc2->superCluster().isNonnull()   && (superCluster() == rc2->superCluster())) return true;
        if (rc2->caloTower().isNonnull()      && (caloTower() == rc2->caloTower())) return true;
   }
   return false;
}
