/** \class HLTTrackMETProducer
 *
 * See header file for documentation
 *
 *  \author Steven Lowette
 *
 */

#include "HLTrigger/JetMET/interface/HLTTrackMETProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/Common/interface/Handle.h"


// Constructor
HLTTrackMETProducer::HLTTrackMETProducer(const edm::ParameterSet & iConfig) :
  usePt_                  ( iConfig.getParameter<bool>("usePt") ),
  useTracks_              ( iConfig.getParameter<bool>("useTracks") ),
  usePFRecTracks_         ( iConfig.getParameter<bool>("usePFRecTracks") ),
  usePFCandidatesCharged_ ( iConfig.getParameter<bool>("usePFCandidatesCharged") ),
  usePFCandidates_        ( iConfig.getParameter<bool>("usePFCandidates") ),
  excludePFMuons_         ( iConfig.getParameter<bool>("excludePFMuons") ),
  minNJet_                ( iConfig.getParameter<int>("minNJet") ),
  minPtJet_               ( iConfig.getParameter<double>("minPtJet") ),
  maxEtaJet_              ( iConfig.getParameter<double>("maxEtaJet") ),
  jetsLabel_              ( iConfig.getParameter<edm::InputTag>("jetsLabel") ),
  tracksLabel_            ( iConfig.getParameter<edm::InputTag>("tracksLabel") ),
  pfRecTracksLabel_       ( iConfig.getParameter<edm::InputTag>("pfRecTracksLabel") ),
  pfCandidatesLabel_      ( iConfig.getParameter<edm::InputTag>("pfCandidatesLabel") ) {
    m_theJetToken = consumes<edm::View<reco::Jet>>(jetsLabel_);
    m_theTrackToken = consumes<reco::TrackCollection>(tracksLabel_);
    m_theRecTrackToken = consumes<reco::PFRecTrackCollection>(pfRecTracksLabel_);
    m_thePFCandidateToken = consumes<reco::PFCandidateCollection>(pfCandidatesLabel_);

    // Register the products
    produces<reco::METCollection>();
}

// Destructor
HLTTrackMETProducer::~HLTTrackMETProducer() {}

// Fill descriptions
void HLTTrackMETProducer::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
    // Current default is for hltPFMET
    edm::ParameterSetDescription desc;
    desc.add<bool>("usePt", true);
    desc.add<bool>("useTracks", false);
    desc.add<bool>("usePFRecTracks", false);
    desc.add<bool>("usePFCandidatesCharged", true);
    desc.add<bool>("usePFCandidates", false);
    desc.add<bool>("excludePFMuons", false);
    desc.add<int>("minNJet",0);
    desc.add<double>("minPtJet", 0.);
    desc.add<double>("maxEtaJet", 999.);
    desc.add<edm::InputTag>("jetsLabel", edm::InputTag("hltAntiKT4PFJets"));
    desc.add<edm::InputTag>("tracksLabel",  edm::InputTag("hltL3Muons"));
    desc.add<edm::InputTag>("pfRecTracksLabel",  edm::InputTag("hltLightPFTracks"));
    desc.add<edm::InputTag>("pfCandidatesLabel",  edm::InputTag("hltParticleFlow"));
    descriptions.add("hltTrackMETProducer", desc);
}

// Produce the products
void HLTTrackMETProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

    // Create a pointer to the products
    std::auto_ptr<reco::METCollection> result(new reco::METCollection());

    if (pfCandidatesLabel_.label() == "")
        excludePFMuons_ = false;

    bool useJets = !useTracks_ && !usePFRecTracks_ && !usePFCandidatesCharged_ && !usePFCandidates_;
    if (!useJets) {
        minNJet_ = 0;
    }

    edm::Handle<reco::JetView> jets;
    if (useJets) iEvent.getByToken(m_theJetToken, jets);

    edm::Handle<reco::TrackCollection> tracks;
    if (useTracks_) iEvent.getByToken(m_theTrackToken, tracks);

    edm::Handle<reco::PFRecTrackCollection> pfRecTracks;
    if (usePFRecTracks_) iEvent.getByToken(m_theRecTrackToken, pfRecTracks);

    edm::Handle<reco::PFCandidateCollection> pfCandidates;
    if (excludePFMuons_ || usePFCandidatesCharged_ || usePFCandidates_)
        iEvent.getByToken(m_thePFCandidateToken, pfCandidates);

    int nj = 0;
    double sumet = 0., mhx = 0., mhy = 0.;

    if (useJets && jets->size() > 0) {
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

    } else if (useTracks_ && tracks->size() > 0) {
        for (reco::TrackCollection::const_iterator j = tracks->begin(); j != tracks->end(); ++j) {
            double pt = j->pt();
            double px = j->px();
            double py = j->py();
            double eta = j->eta();

            if (pt > minPtJet_ && std::abs(eta) < maxEtaJet_) {
                mhx -= px;
                mhy -= py;
                sumet += pt;
                ++nj;
            }
        }

    } else if (usePFRecTracks_ && pfRecTracks->size() > 0) {
        for (reco::PFRecTrackCollection::const_iterator j = pfRecTracks->begin(); j != pfRecTracks->end(); ++j) {
            double pt = j->trackRef()->pt();
            double px = j->trackRef()->px();
            double py = j->trackRef()->py();
            double eta = j->trackRef()->eta();

            if (pt > minPtJet_ && std::abs(eta) < maxEtaJet_) {
                mhx -= px;
                mhy -= py;
                sumet += pt;
                ++nj;
            }
        }

    } else if ((usePFCandidatesCharged_ || usePFCandidates_) && pfCandidates->size() > 0) {
        for (reco::PFCandidateCollection::const_iterator j = pfCandidates->begin(); j != pfCandidates->end(); ++j) {
            if (usePFCandidatesCharged_ && j->charge() == 0)  continue;
            double pt = j->pt();
            double px = j->px();
            double py = j->py();
            double eta = j->eta();

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
