/** \class HLTMhtProducer
 *
 * See header file for documentation
 *
 *  \author Steven Lowette
 *
 */

#include "HLTrigger/JetMET/interface/HLTMhtProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/Common/interface/Handle.h"


// Constructor
HLTMhtProducer::HLTMhtProducer(const edm::ParameterSet & iConfig) :
  usePt_                  ( iConfig.getParameter<bool>("usePt") ),
  excludePFMuons_         ( iConfig.getParameter<bool>("excludePFMuons") ),
  minNJet_                ( iConfig.getParameter<int>("minNJet") ),
  minPtJet_               ( iConfig.getParameter<double>("minPtJet") ),
  maxEtaJet_              ( iConfig.getParameter<double>("maxEtaJet") ),
  jetsLabel_              ( iConfig.getParameter<edm::InputTag>("jetsLabel") ),
  pfCandidatesLabel_      ( iConfig.getParameter<edm::InputTag>("pfCandidatesLabel") ) {
    m_theJetToken = consumes<edm::View<reco::Jet>>(jetsLabel_);
    if (pfCandidatesLabel_.label() == "") excludePFMuons_ = false;
    if (excludePFMuons_) m_thePFCandidateToken = consumes<reco::PFCandidateCollection>(pfCandidatesLabel_);

    // Register the products
    produces<reco::METCollection>();
}

// Destructor
HLTMhtProducer::~HLTMhtProducer() {}

// Fill descriptions
void HLTMhtProducer::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
    // Current default is for hltPFMET
    edm::ParameterSetDescription desc;
    desc.add<bool>("usePt", true);
    desc.add<bool>("excludePFMuons", false);
    desc.add<int>("minNJet",0);
    desc.add<double>("minPtJet", 0.);
    desc.add<double>("maxEtaJet", 999.);
    desc.add<edm::InputTag>("jetsLabel", edm::InputTag("hltAntiKT4PFJets"));
    desc.add<edm::InputTag>("pfCandidatesLabel",  edm::InputTag("hltParticleFlow"));
    descriptions.add("hltMhtProducer", desc);
}

// Produce the products
void HLTMhtProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

    // Create a pointer to the products
    std::auto_ptr<reco::METCollection> result(new reco::METCollection());

    edm::Handle<reco::JetView> jets;
    iEvent.getByToken(m_theJetToken, jets);

    edm::Handle<reco::PFCandidateCollection> pfCandidates;
    if (excludePFMuons_)
        iEvent.getByToken(m_thePFCandidateToken, pfCandidates);

    int nj = 0;
    double sumet = 0., mhx = 0., mhy = 0.;

    if (jets->size() > 0) {
        for(reco::JetView::const_iterator j = jets->begin(); j != jets->end(); ++j) {
            double pt = usePt_ ? j->pt() : j->et();
            double eta = j->eta();
            double phi = j->phi();
            double px = usePt_ ? j->px() : j->et() * cos(phi);
            double py = usePt_ ? j->py() : j->et() * sin(phi);

            if (pt > minPtJet_ && std::abs(eta) < maxEtaJet_) {
                mhx -= px;
                mhy -= py;
                sumet += pt;
                ++nj;
            }
        }
    }

    if (excludePFMuons_) {
        for (reco::PFCandidateCollection::const_iterator j = pfCandidates->begin(); j != pfCandidates->end(); ++j) {
            if (std::abs(j->pdgId()) == 13) {
                mhx += j->px();
                mhy += j->py();
            }
        }
    }

    if (nj < minNJet_) { sumet = 0; mhx = 0; mhy = 0; }

    reco::MET::LorentzVector p4(mhx, mhy, 0, sqrt(mhx*mhx + mhy*mhy));
    reco::MET::Point vtx(0, 0, 0);
    reco::MET mht(sumet, p4, vtx);
    result->push_back(mht);

    // Put the products into the Event
    iEvent.put(result);
}
