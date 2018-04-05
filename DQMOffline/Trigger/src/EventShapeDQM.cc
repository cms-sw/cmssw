#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DQMOffline/Trigger/interface/EventShapeDQM.h"

EventShapeDQM::EventShapeDQM(const edm::ParameterSet& ps)
{
	triggerResults_ = consumes<edm::TriggerResults>(ps.getParameter<edm::InputTag>("triggerResults"));
	theEPCollection_ = consumes<reco::EvtPlaneCollection>(ps.getParameter<edm::InputTag>("EPlabel"));
	triggerPath_ = ps.getParameter<std::string>("triggerPath");

	order_ = ps.getParameter<int>("order");
	EPidx_ = ps.getParameter<int>("EPidx");
	EPlvl_ = ps.getParameter<int>("EPlvl");
}

EventShapeDQM::~EventShapeDQM() = default;


void EventShapeDQM::bookHistograms(DQMStore::IBooker & ibooker_, edm::Run const &, edm::EventSetup const &)
{
	ibooker_.cd();;
	ibooker_.setCurrentFolder("HLT/HI/" + triggerPath_);

	h_Q = ibooker_.book1D("hQn", Form("Q%i;Q%i", order_, order_), 500, 0, 0.5);

	ibooker_.cd();
}

void EventShapeDQM::analyze(edm::Event const& e, edm::EventSetup const& eSetup)
{

	edm::Handle<edm::TriggerResults> hltresults;
	e.getByToken(triggerResults_,hltresults);
	if(!hltresults.isValid())
	{
		return;
	}

	bool hasFired = false;
	const edm::TriggerNames& trigNames = e.triggerNames(*hltresults);
	unsigned int numTriggers = trigNames.size();
	for( unsigned int hltIndex=0; hltIndex<numTriggers; ++hltIndex ) {
		if (trigNames.triggerName(hltIndex).find(triggerPath_) != std::string::npos && hltresults->wasrun(hltIndex) && hltresults->accept(hltIndex)){
			hasFired = true;
		}
	}

	edm::Handle<reco::EvtPlaneCollection> ep_;
	e.getByToken(theEPCollection_, ep_);
	if ( !ep_.isValid() ) {
		return;
	}

	if ( hasFired ) {
		h_Q->Fill( (*ep_)[EPidx_].vn(EPlvl_) );
	}

}

DEFINE_FWK_MODULE(EventShapeDQM);
