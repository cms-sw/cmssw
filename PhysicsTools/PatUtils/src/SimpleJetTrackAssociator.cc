
#include "PhysicsTools/PatUtils/interface/SimpleJetTrackAssociator.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include <Math/VectorUtil.h>

helper::SimpleJetTrackAssociator::SimpleJetTrackAssociator() :
   deltaR_(0), nHits_(0), chi2nMax_(0) { }

helper::SimpleJetTrackAssociator::SimpleJetTrackAssociator(const edm::ParameterSet &iCfg) :
        deltaR_(iCfg.getParameter<double>("deltaR")),
        nHits_(iCfg.getParameter<int32_t>("minHits")),
        chi2nMax_(iCfg.getParameter<double>("maxNormChi2")) 
{
        // nothing to be done here; 
}

void
helper::SimpleJetTrackAssociator::associate(const math::XYZVector &dir, 
                        const edm::Handle<reco::TrackCollection> &in, reco::TrackRefVector &out) {
        for (size_t i = 0, n = in->size(); i < n; i++) {
                reco::TrackRef t(in,i);
                if ((t->numberOfValidHits() < nHits_) || (t->normalizedChi2() > chi2nMax_)) continue;
                if (ROOT::Math::VectorUtil::DeltaR(dir, t->momentum()) < deltaR_) {
                        out.push_back(t);
                }
        }
}
