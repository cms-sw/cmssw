/** \class HLTHtMhtProducer
 *
 * See header file for documentation
 *
 *  \author Steven Lowette
 *
 */

#include "HLTrigger/JetMET/interface/HLTHtMhtProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/Common/interface/Handle.h"


// Constructor
HLTHtMhtProducer::HLTHtMhtProducer(const edm::ParameterSet & iConfig) :
  usePt_                  ( iConfig.getParameter<bool>("usePt") ),
  excludePFMuons_         ( iConfig.getParameter<bool>("excludePFMuons") ),
  minNJetHt_              ( iConfig.getParameter<int>("minNJetHt") ),
  minNJetMht_             ( iConfig.getParameter<int>("minNJetMht") ),
  minPtJetHt_             ( iConfig.getParameter<double>("minPtJetHt") ),
  minPtJetMht_            ( iConfig.getParameter<double>("minPtJetMht") ),
  maxEtaJetHt_            ( iConfig.getParameter<double>("maxEtaJetHt") ),
  maxEtaJetMht_           ( iConfig.getParameter<double>("maxEtaJetMht") ),
  jetsLabel_              ( iConfig.getParameter<edm::InputTag>("jetsLabel") ),
  pfCandidatesLabel_      ( iConfig.getParameter<edm::InputTag>("pfCandidatesLabel") ) {
    m_theJetToken = consumes<edm::View<reco::Jet>>(jetsLabel_);
    m_thePFCandidateToken = consumes<reco::PFCandidateCollection>(pfCandidatesLabel_);

    // Register the products
    produces<reco::METCollection>();
}

// Destructor
HLTHtMhtProducer::~HLTHtMhtProducer() {}

// Fill descriptions
void HLTHtMhtProducer::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
    // Current default is for hltHtMht
    edm::ParameterSetDescription desc;
    desc.add<bool>("usePt", false);
    desc.add<bool>("excludePFMuons", false);
    desc.add<int>("minNJetHt", 0);
    desc.add<int>("minNJetMht", 0);
    desc.add<double>("minPtJetHt", 40.);
    desc.add<double>("minPtJetMht", 30.);
    desc.add<double>("maxEtaJetHt", 3.);
    desc.add<double>("maxEtaJetMht", 5.);
    desc.add<edm::InputTag>("jetsLabel", edm::InputTag("hltCaloJetL1FastJetCorrected"));
    desc.add<edm::InputTag>("pfCandidatesLabel",  edm::InputTag("hltParticleFlow"));
    descriptions.add("hltHtMhtProducer", desc);
}

// Produce the products
void HLTHtMhtProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

    // Create a pointer to the products
    std::auto_ptr<reco::METCollection> result(new reco::METCollection());

    if (pfCandidatesLabel_.label() == "")
        excludePFMuons_ = false;

    edm::Handle<reco::JetView> jets;
    iEvent.getByToken(m_theJetToken, jets);

    edm::Handle<reco::PFCandidateCollection> pfCandidates;
    if (excludePFMuons_)
        iEvent.getByToken(m_thePFCandidateToken, pfCandidates);

    int nj_ht = 0, nj_mht = 0;
    double ht = 0., mhx = 0., mhy = 0.;

    if (jets->size() > 0) {
        for(reco::JetView::const_iterator j = jets->begin(); j != jets->end(); ++j) {
            double pt = usePt_ ? j->pt() : j->et();
            double eta = j->eta();
            double phi = j->phi();
            double px = usePt_ ? j->px() : j->et() * cos(phi);
            double py = usePt_ ? j->py() : j->et() * sin(phi);

            if (pt > minPtJetHt_ && std::abs(eta) < maxEtaJetHt_) {
                ht += pt;
                ++nj_ht;
            }

            if (pt > minPtJetMht_ && std::abs(eta) < maxEtaJetMht_) {
                mhx -= px;
                mhy -= py;
                ++nj_mht;
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

    if (nj_ht  < minNJetHt_ ) { ht = 0; }
    if (nj_mht < minNJetMht_) { mhx = 0; mhy = 0; }

    reco::MET::LorentzVector p4(mhx, mhy, 0, sqrt(mhx*mhx + mhy*mhy));
    reco::MET::Point vtx(0, 0, 0);
    reco::MET htmht(ht, p4, vtx);
    result->push_back(htmht);

    // Put the products into the Event
    iEvent.put(result);
}
