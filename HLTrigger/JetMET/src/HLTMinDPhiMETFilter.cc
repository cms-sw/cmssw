/** \class  HLTMinDPhiMETFilter
 *
 *  See header file for more information.
 *  
 *  \author a Jet/MET person
 *
 */

#include "HLTrigger/JetMET/interface/HLTMinDPhiMETFilter.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
//#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
//#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/Math/interface/deltaPhi.h"


// Constructor
HLTMinDPhiMETFilter::HLTMinDPhiMETFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig),
  usePt_          (iConfig.getParameter<bool>("usePt")),
  //excludePFMuons_ (iConfig.getParameter<bool>("excludePFMuons")),
  triggerType_    (iConfig.getParameter<int>("triggerType")),
  maxNJets_       (iConfig.getParameter<int>("maxNJets")),
  minPt_          (iConfig.getParameter<double>("minPt")),
  maxEta_         (iConfig.getParameter<double>("maxEta")),
  minDPhi_        (iConfig.getParameter<double>("minDPhi")),
  metLabel_       (iConfig.getParameter<edm::InputTag>("metLabel")),
  calometLabel_   (iConfig.getParameter<edm::InputTag>("calometLabel")),
  jetsLabel_      (iConfig.getParameter<edm::InputTag>("jetsLabel")) {
    m_theMETToken = consumes<reco::METCollection>(metLabel_);
    m_theCaloMETToken = consumes<reco::CaloMETCollection>(calometLabel_);
    m_theJetToken = consumes<reco::JetView>(jetsLabel_);
}

// Destructor
HLTMinDPhiMETFilter::~HLTMinDPhiMETFilter() {}

// Fill descriptions
void HLTMinDPhiMETFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    makeHLTFilterDescription(desc);
    desc.add<bool>("usePt", true);
    //desc.add<bool>("excludePFMuons", false);
    desc.add<int>("triggerType", trigger::TriggerJet);
    desc.add<int>("maxNJets", 2);
    desc.add<double>("minPt", 30.);
    desc.add<double>("maxEta", 2.5);
    desc.add<double>("minDPhi", 0.5);
    desc.add<edm::InputTag>("metLabel", edm::InputTag("hltPFMETProducer"));
    desc.add<edm::InputTag>("calometLabel", edm::InputTag(""));
    desc.add<edm::InputTag>("jetsLabel", edm::InputTag("hltAK4PFJetL1FastL2L3Corrected"));
    descriptions.add("hltMinDPhiMETFilter", desc);
}

// Make filter decision
bool HLTMinDPhiMETFilter::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) const {

    // The filter object
    if (saveTags()) filterproduct.addCollectionTag(jetsLabel_);

    bool usePFMET = (metLabel_.label() != "") || (calometLabel_.label() == "");

    edm::Handle<reco::METCollection> mets;
    edm::Handle<reco::CaloMETCollection> calomets;
    if (usePFMET) {
        iEvent.getByToken(m_theMETToken, mets);
    } else {
        iEvent.getByToken(m_theMETToken, calomets);
    }

    edm::Handle<reco::JetView> jets;  // assume to be sorted by pT
    iEvent.getByToken(m_theJetToken, jets);

    double minDPhi = 3.141593;
    int nJets = 0;  // nJets counts all jets in the events, not only those that pass pt, eta requirements

    if (jets->size() > 0 &&
        ((usePFMET ? mets->size() : calomets->size()) > 0) ) {
        double metphi = usePFMET ? mets->front().phi() : calomets->front().phi();
        for (reco::JetView::const_iterator j = jets->begin(); j != jets->end(); ++j) {
            if (nJets >= maxNJets_)
                break;

            double pt = usePt_ ? j->pt() : j->et();
            double eta = j->eta();
            double phi = j->phi();
            if (pt > minPt_ && std::abs(eta) < maxEta_) {
                double dPhi = std::abs(reco::deltaPhi(metphi, phi));
                if (minDPhi > dPhi) {
                    minDPhi = dPhi;
                }

                // Not sure what to save, since an event quantity is used
                //reco::JetBaseRef ref(jets, distance(jets->begin(), j));
                //filterproduct.addObject(triggerType_, ref);
            }

            ++nJets;
        }
    }

    bool accept(minDPhi > minDPhi_);

    return accept;
}
