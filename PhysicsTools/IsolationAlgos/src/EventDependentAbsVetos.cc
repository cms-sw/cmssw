
#include "PhysicsTools/IsolationAlgos/interface/EventDependentAbsVetos.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Math/interface/deltaR.h"

bool 
reco::isodeposit::OtherCandidatesDeltaRVeto::veto(double eta, double phi, float value) const 
{
    for (std::vector<Direction>::const_iterator it = items_.begin(), ed = items_.end(); it != ed; ++it) {
        if (::deltaR2(it->eta(), it->phi(), eta, phi) < deltaR2_) return true;
    }
    return false;
}

void
reco::isodeposit::OtherCandidatesDeltaRVeto::setEvent(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
    items_.clear();
    edm::Handle<edm::View<reco::Candidate> > candidates;
    iEvent.getByLabel(src_, candidates);
    for (edm::View<reco::Candidate>::const_iterator it = candidates->begin(), ed = candidates->end(); it != ed; ++it) {
        items_.push_back(Direction(it->eta(), it->phi()));
    }
}

bool 
reco::isodeposit::OtherCandVeto::veto(double eta, double phi, float value) const 
{
    for (std::vector<Direction>::const_iterator it = items_.begin(), ed = items_.end(); it != ed; ++it) {
        veto_->centerOn(it->eta(), it->phi());
        if ( veto_->veto(eta,phi,value) ) return true;
    }
    return false;
}

void
reco::isodeposit::OtherCandVeto::setEvent(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
    items_.clear();
    edm::Handle<edm::View<reco::Candidate> > candidates;
    iEvent.getByLabel(src_, candidates);
    for (edm::View<reco::Candidate>::const_iterator it = candidates->begin(), ed = candidates->end(); it != ed; ++it) {
        items_.push_back(Direction(it->eta(), it->phi()));
    }
}

