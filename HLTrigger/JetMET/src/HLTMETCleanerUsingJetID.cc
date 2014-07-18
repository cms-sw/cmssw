/** \class HLTMETCleanerUsingJetID
 *
 * See header file for more information.
 *
 *  \author a Jet/MET person
 *
 */

#include "HLTrigger/JetMET/interface/HLTMETCleanerUsingJetID.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"


// Constructor
HLTMETCleanerUsingJetID::HLTMETCleanerUsingJetID(const edm::ParameterSet& iConfig)
      : usePt_         (iConfig.getParameter<bool>("usePt")),
        minPt_         (iConfig.getParameter<double>("minPt")),
        maxEta_        (iConfig.getParameter<double>("maxEta")),
        metLabel_      (iConfig.getParameter<edm::InputTag>("metLabel")),
        jetsLabel_     (iConfig.getParameter<edm::InputTag>("jetsLabel")),
        goodJetsLabel_ (iConfig.getParameter<edm::InputTag>("goodJetsLabel")) {
    m_theMETToken = consumes<reco::CaloMETCollection>(metLabel_);
    m_theJetToken = consumes<reco::CaloJetCollection>(jetsLabel_);
    m_theGoodJetToken = consumes<reco::CaloJetCollection>(goodJetsLabel_);

    // Register the products
    produces<reco::CaloMETCollection>();
}

// Destructor
HLTMETCleanerUsingJetID::~HLTMETCleanerUsingJetID() {}

// Fill descriptions
void HLTMETCleanerUsingJetID::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<bool>("usePt", false);
    desc.add<double>("minPt", 20.);
    desc.add<double>("maxEta", 5.);
    desc.add<edm::InputTag>("metLabel", edm::InputTag("hltMet"));
    desc.add<edm::InputTag>("jetsLabel", edm::InputTag("hltAntiKT4CaloJets"));
    desc.add<edm::InputTag>("goodJetsLabel", edm::InputTag("hltCaloJetIDPassed"));
    descriptions.add("hltMETCleanerUsingJetID",desc);
}

// Produce the products
void HLTMETCleanerUsingJetID::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

    // Create a pointer to the products
    std::auto_ptr<reco::CaloMETCollection> result(new reco::CaloMETCollection);

    edm::Handle<reco::CaloMETCollection> met;
    edm::Handle<reco::CaloJetCollection> jets;
    edm::Handle<reco::CaloJetCollection> goodJets;

    iEvent.getByToken(m_theMETToken, met);
    iEvent.getByToken(m_theJetToken, jets);
    iEvent.getByToken(m_theGoodJetToken, goodJets);

    double mex_jets = 0.;
    double mey_jets = 0.;
    double sumet_jets = 0.;
    if (jets->size() > 0 ) {
        for(reco::CaloJetCollection::const_iterator j = jets->begin(); j != jets->end(); ++j) {
            double pt = usePt_ ? j->pt() : j->et();
            double eta = j->eta();
            double phi = j->phi();
            double px = usePt_ ? j->px() : j->et() * cos(phi);
            double py = usePt_ ? j->py() : j->et() * sin(phi);

            if (pt > minPt_ && std::abs(eta) < maxEta_) {
                mex_jets -= px;
                mey_jets -= py;
                sumet_jets += pt;
            }
        }
    }

    double mex_goodJets = 0.;
    double mey_goodJets = 0.;
    double sumet_goodJets = 0.;
    if (goodJets->size() > 0) {
        for(reco::CaloJetCollection::const_iterator j = goodJets->begin(); j != goodJets->end(); ++j) {
            double pt = usePt_ ? j->pt() : j->et();
            double eta = j->eta();
            double phi = j->phi();
            double px = usePt_ ? j->px() : j->pt() * cos(phi);
            double py = usePt_ ? j->py() : j->pt() * sin(phi);

            if (pt > minPt_ && std::abs(eta) < maxEta_) {
                mex_goodJets -= px;
                mey_goodJets -= py;
                sumet_goodJets += pt;
            }
        }
    }

    if (met->size() > 0) {
        double mex_diff = mex_goodJets - mex_jets;
        double mey_diff = mey_goodJets - mey_jets;
        //double sumet_diff = sumet_goodJets - sumet_jets;  // cannot set sumet...
        reco::Candidate::LorentzVector p4_diff(mex_diff, mey_diff, 0, sqrt(mex_diff*mex_diff + mey_diff*mey_diff));

        reco::CaloMET cleanmet = met->front();
        cleanmet.setP4(cleanmet.p4() + p4_diff);
        result->push_back(cleanmet);
    }

    iEvent.put( result );
}
