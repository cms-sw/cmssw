#ifndef EventShapeDQM_H
#define EventShapeDQM_H
//event
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

//DQM
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

//EP
#include "DataFormats/HeavyIonEvent/interface/EvtPlane.h"

// Trigger
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

class EventShapeDQM: public DQMEDAnalyzer{
public:
	EventShapeDQM(const edm::ParameterSet& ps);
	virtual ~EventShapeDQM();

protected:
	void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
	void analyze(edm::Event const& e, edm::EventSetup const& eSetup);

private:
	edm::EDGetTokenT<edm::TriggerResults> triggerResults_;
	edm::EDGetTokenT<reco::EvtPlaneCollection> theEPCollection_;

	std::string triggerPath_;
	int order_;
	int EPidx_;
	int EPlvl_;

	// histo
	MonitorElement* h_Q;
};

#endif
