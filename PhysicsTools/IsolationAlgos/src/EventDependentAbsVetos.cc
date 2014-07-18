
#include "PhysicsTools/IsolationAlgos/interface/EventDependentAbsVetos.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
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
    iEvent.getByToken(src_, candidates);
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
    iEvent.getByToken(src_, candidates);
    for (edm::View<reco::Candidate>::const_iterator it = candidates->begin(), ed = candidates->end(); it != ed; ++it) {
        items_.push_back(Direction(it->eta(), it->phi()));
    }
}

bool
reco::isodeposit::OtherJetConstituentsDeltaRVeto::veto(double eta, double phi, float value) const
{
    for (std::vector<Direction>::const_iterator it = items_.begin(), ed = items_.end(); it != ed; ++it) {
        if (::deltaR2(it->eta(), it->phi(), eta, phi) < dR2constituent_) return true;
    }
    return false;
}

void
reco::isodeposit::OtherJetConstituentsDeltaRVeto::setEvent(const edm::Event& evt, const edm::EventSetup& es)
{
    //std::cout << "<OtherJetConstituentsDeltaRVeto::setEvent>:" << std::endl;
    evt_ = &evt;
}
void
reco::isodeposit::OtherJetConstituentsDeltaRVeto::initialize()
{
    //std::cout << "<OtherJetConstituentsDeltaRVeto::initialize>:" << std::endl;
    //std::cout << " vetoDir: eta = " << vetoDir_.eta() << ", phi = " << vetoDir_.phi() << std::endl;
    assert(evt_);
    items_.clear();
    edm::Handle<reco::PFJetCollection> jets;
    evt_->getByToken(srcJets_, jets);
    edm::Handle<JetToPFCandidateAssociation> jetToPFCandMap;
    evt_->getByToken(srcPFCandAssocMap_, jetToPFCandMap);
    double dR2min = dR2jet_;
    reco::PFJetRef matchedJet;
    size_t numJets = jets->size();
    for ( size_t jetIndex = 0; jetIndex < numJets; ++jetIndex ) {
      reco::PFJetRef jet(jets, jetIndex);
      double dR2 = ::deltaR2(vetoDir_.eta(), vetoDir_.phi(), jet->eta(), jet->phi());
      //std::cout << "jet #" << jetIndex << ": Pt = " << jet->pt() << ", eta = " << jet->eta() << ", phi = " << jet->phi() << " (dR = " << sqrt(dR2) << ")" << std::endl;
      if ( dR2 < dR2min ) {
	matchedJet = jet;
	dR2min = dR2;
      }
    }
    if ( matchedJet.isNonnull() ) {
      edm::RefVector<reco::PFCandidateCollection> pfCandsMappedToJet = (*jetToPFCandMap)[matchedJet];
      int idx = 0;
      for ( edm::RefVector<reco::PFCandidateCollection>::const_iterator pfCand = pfCandsMappedToJet.begin();
	    pfCand != pfCandsMappedToJet.end(); ++pfCand ) {
	//std::cout << "pfCand #" << idx << ": Pt = " << (*pfCand)->pt() << ", eta = " << (*pfCand)->eta() << ", phi = " << (*pfCand)->phi() << std::endl;
	items_.push_back(Direction((*pfCand)->eta(), (*pfCand)->phi()));
	++idx;
      }
    }
}

void
reco::isodeposit::OtherJetConstituentsDeltaRVeto::centerOn(double eta, double phi)
{
    //std::cout << "<OtherJetConstituentsDeltaRVeto::centerOn>:" << std::endl;
    //std::cout << " eta = " << eta << std::endl;
    //std::cout << " phi = " << phi << std::endl;
    vetoDir_ = Direction(eta,phi);
    initialize();
}
