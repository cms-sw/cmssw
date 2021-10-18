/** \class HLTCaloJetTimingProducer
 *
 * See header file for documentation
 *
 *  \author Matthew Citron
 *
 */

#include "HLTrigger/JetMET/interface/HLTCaloJetTimingProducer.h"

//Constructor
HLTCaloJetTimingProducer::HLTCaloJetTimingProducer(const edm::ParameterSet& iConfig)
{
    produces<edm::ValueMap<float>>("");
    produces<edm::ValueMap<unsigned int>>("jetCellsForTiming");
    produces<edm::ValueMap<float>>("jetEcalEtForTiming");
    jetLabel_= iConfig.getParameter<edm::InputTag>("jets");
    ecalEBLabel_= iConfig.getParameter<edm::InputTag>("ebRecHitsColl");
    ecalEELabel_= iConfig.getParameter<edm::InputTag>("eeRecHitsColl");
    barrelOnly_ = iConfig.getParameter<bool>("barrelOnly");
    jetInputToken = consumes<std::vector<reco::CaloJet>>(jetLabel_);
    ecalRecHitsEBToken = consumes<edm::SortedCollection<EcalRecHit,edm::StrictWeakOrdering<EcalRecHit>>>(ecalEBLabel_);
    ecalRecHitsEEToken = consumes<edm::SortedCollection<EcalRecHit,edm::StrictWeakOrdering<EcalRecHit>>>(ecalEELabel_);
}

//Producer
void HLTCaloJetTimingProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    using namespace edm;
    std::vector<float> jetTimings;
    std::vector<unsigned int> jetCellsForTiming;
    std::vector<float> jetEcalEtForTiming;
    Handle<reco::CaloJetCollection> jets;
    iEvent.getByToken(jetInputToken, jets); 
    Handle<edm::SortedCollection<EcalRecHit,edm::StrictWeakOrdering<EcalRecHit>>> ecalRecHitsEB;
    iEvent.getByToken(ecalRecHitsEBToken,ecalRecHitsEB);
    Handle<edm::SortedCollection<EcalRecHit,edm::StrictWeakOrdering<EcalRecHit>>> ecalRecHitsEE;
    iEvent.getByToken(ecalRecHitsEEToken,ecalRecHitsEE);
    edm::ESHandle<CaloGeometry> pG; 
    iSetup.get<CaloGeometryRecord>().get(pG); 
    for (auto const& c : *jets) {
	int iCell = -1;
	float weightedTimeCell = 0;
	float totalEmEnergyCell = 0;
	unsigned int nCells = 0;
	for (EcalRecHitCollection::const_iterator i=ecalRecHitsEB->begin(); i!=ecalRecHitsEB->end(); i++) {
	    iCell++;
	    if ((i->checkFlag(EcalRecHit::kSaturated) || i->checkFlag(EcalRecHit::kLeadingEdgeRecovered) || i->checkFlag(EcalRecHit::kPoorReco) || i->checkFlag(EcalRecHit::kWeird) || i->checkFlag(EcalRecHit::kDiWeird))) continue;
            if (i->energy() < 0.5) continue;
            if (i->timeError() <= 0. || i->timeError() > 100) continue;
            if (i->time() < -12.5 || i->time() > 12.5) continue;
            GlobalPoint p=pG->getPosition(i->detid());
            if (reco::deltaR(c,p) > 0.4) continue;
	    weightedTimeCell += i->time()*i->energy()*sin(p.theta());
	    totalEmEnergyCell += i->energy()*sin(p.theta());
            nCells++;
	}
	iCell = -1;
	if (!barrelOnly_){
	    for (EcalRecHitCollection::const_iterator i=ecalRecHitsEE->begin(); i!=ecalRecHitsEE->end(); i++) {
		iCell++;
		if ((i->checkFlag(EcalRecHit::kSaturated) || i->checkFlag(EcalRecHit::kLeadingEdgeRecovered) || i->checkFlag(EcalRecHit::kPoorReco) || i->checkFlag(EcalRecHit::kWeird) || i->checkFlag(EcalRecHit::kDiWeird))) continue;
		if (i->energy() < 0.5) continue;
		if (i->timeError() <= 0. || i->timeError() > 100) continue;
		if (i->time() < -12.5 || i->time() > 12.5) continue;
		GlobalPoint p=pG->getPosition(i->detid());
		if (reco::deltaR(c,p) > 0.4) continue;
		weightedTimeCell += i->time()*i->energy()*sin(p.theta());
		totalEmEnergyCell += i->energy()*sin(p.theta());
		nCells++;
	    }
	}
	//If there is at least one ecal cell passing selection
	// calculate timing
	if (totalEmEnergyCell > 0){
	    jetTimings.push_back(weightedTimeCell/totalEmEnergyCell);
	    jetEcalEtForTiming.push_back(totalEmEnergyCell);
	    jetCellsForTiming.push_back(nCells);
	} 
	else{
	    jetTimings.push_back(-50);
	    jetEcalEtForTiming.push_back(totalEmEnergyCell);
	    jetCellsForTiming.push_back(nCells);
	}
    }
    std::unique_ptr<edm::ValueMap<float> > jetTimings_out(new edm::ValueMap<float>());
    edm::ValueMap<float>::Filler jetTimings_filler(*jetTimings_out);
    jetTimings_filler.insert(jets, jetTimings.begin(), jetTimings.end());
    jetTimings_filler.fill();
    iEvent.put(std::move(jetTimings_out), "");

    std::unique_ptr<edm::ValueMap<float> > jetEcalEtForTiming_out(new edm::ValueMap<float>());
    edm::ValueMap<float>::Filler jetEcalEtForTiming_filler(*jetEcalEtForTiming_out);
    jetEcalEtForTiming_filler.insert(jets, jetEcalEtForTiming.begin(), jetEcalEtForTiming.end());
    jetEcalEtForTiming_filler.fill();
    iEvent.put(std::move(jetEcalEtForTiming_out), "jetEcalEtForTiming");

    std::unique_ptr<edm::ValueMap<unsigned int> > jetCellsForTiming_out(new edm::ValueMap<unsigned int>());
    edm::ValueMap<unsigned int>::Filler jetCellsForTiming_filler(*jetCellsForTiming_out);
    jetCellsForTiming_filler.insert(jets, jetCellsForTiming.begin(), jetCellsForTiming.end());
    jetCellsForTiming_filler.fill();
    iEvent.put(std::move(jetCellsForTiming_out), "jetCellsForTiming");
}

// Fill descriptions
void HLTCaloJetTimingProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("jets", edm::InputTag(""));
    desc.add<bool>("barrelOnly", false);
    desc.add<edm::InputTag>("ebRecHitsColl", edm::InputTag("hltEcalRecHit","EcalRecHitsEB"));
    desc.add<edm::InputTag>("eeRecHitsColl", edm::InputTag("hltEcalRecHit","EcalRecHitsEE"));
    descriptions.add("caloJetTimingProducer", desc);
}
