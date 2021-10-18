/** \class HLTCaloJetTimingFilter
 *
 * See header file for documentation
 *
 *  \author Matthew Citron
 *
 */

#include "HLTrigger/JetMET/interface/HLTCaloJetTimingFilter.h"

//Constructor
HLTCaloJetTimingFilter::HLTCaloJetTimingFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig){
    jetLabel_= iConfig.getParameter<edm::InputTag>("jets");
    jetTimeLabel_= iConfig.getParameter<edm::InputTag>("jetTimes");
    jetCellsForTimingLabel_= iConfig.getParameter<edm::InputTag>("jetCellsForTiming");
    jetEcalEtForTimingLabel_= iConfig.getParameter<edm::InputTag>("jetEcalEtForTiming");
    minJets_= iConfig.getParameter<unsigned int>("minJets");
    jetTimeThresh_ = iConfig.getParameter<double>("jetTimeThresh");
    jetCellsForTimingThresh_ = iConfig.getParameter<unsigned int>("jetCellsForTimingThresh");
    jetEcalEtForTimingThresh_ = iConfig.getParameter<double>("jetEcalEtForTimingThresh");
    minPt_ = iConfig.getParameter<double>("minJetPt");
    jetInputToken = consumes<std::vector<reco::CaloJet>>(jetLabel_);
    jetTimesInputToken = consumes<edm::ValueMap<float>>(jetTimeLabel_);
    jetCellsForTimingInputToken = consumes<edm::ValueMap<unsigned int>>(jetCellsForTimingLabel_);
    jetEcalEtForTimingInputToken = consumes<edm::ValueMap<float>>(jetEcalEtForTimingLabel_);
    //now do what ever initialization is needed
}

//Filter
bool HLTCaloJetTimingFilter::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup,trigger::TriggerFilterObjectWithRefs& filterproduct) const {
    bool accept = false;
    int ijet = 0;
    edm::Handle<reco::CaloJetCollection> jets;
    iEvent.getByToken(jetInputToken, jets);
    edm::Handle<edm::ValueMap<float>> jetTimes;
    iEvent.getByToken(jetTimesInputToken, jetTimes);
    edm::Handle<edm::ValueMap<unsigned int>> jetCellsForTiming;
    iEvent.getByToken(jetCellsForTimingInputToken, jetCellsForTiming);
    edm::Handle<edm::ValueMap<float>> jetEcalEtForTiming;
    iEvent.getByToken(jetEcalEtForTimingInputToken, jetEcalEtForTiming);
    unsigned int njets = 0;
    for (auto const& c : *jets) {
	reco::CaloJetRef calojetref(jets, ijet);
	if((*jetTimes)[calojetref] > jetTimeThresh_ && (*jetEcalEtForTiming)[calojetref] > jetEcalEtForTimingThresh_ && (*jetCellsForTiming)[calojetref] > jetCellsForTimingThresh_ && c.pt() > minPt_) njets++;
	ijet ++;
    }
    accept = njets >= minJets_;
    return accept;
}


// Fill descriptions
void HLTCaloJetTimingFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    makeHLTFilterDescription(desc);
    desc.add<edm::InputTag>("jets", edm::InputTag("hltDisplacedHLTCaloJetCollectionProducerMidPt"));
    desc.add<edm::InputTag>("jetTimes", edm::InputTag("hltDisplacedHLTCaloJetCollectionProducerMidPtTiming"));
    desc.add<edm::InputTag>("jetCellsForTiming", edm::InputTag("hltDisplacedHLTCaloJetCollectionProducerMidPtTiming","jetCellsForTiming"));
    desc.add<edm::InputTag>("jetEcalEtForTiming", edm::InputTag("hltDisplacedHLTCaloJetCollectionProducerMidPtTiming","jetEcalEtForTiming"));
    desc.add<unsigned int>("minJets", 1);
    desc.add<double>("jetTimeThresh", 1.);
    desc.add<unsigned int>("jetCellsForTimingThresh", 5);
    desc.add<double>("jetEcalEtForTimingThresh", 10.);
    desc.add<double>("minJetPt", 40.);
    descriptions.add("caloJetTimingFilter", desc);
}
