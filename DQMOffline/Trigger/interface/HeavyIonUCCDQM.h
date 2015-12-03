#ifndef HeavyIonUCCDQM_H
#define HeavyIonUCCDQM_H
//event
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

//DQM
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

//Centrality
#include "DataFormats/HeavyIonEvent/interface/Centrality.h"
//SiPixelClusters
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"

// Trigger
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

class HeavyIonUCCDQM: public DQMEDAnalyzer{
public:
	HeavyIonUCCDQM(const edm::ParameterSet& ps);
	virtual ~HeavyIonUCCDQM();

protected:
	void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
	void analyze(edm::Event const& e, edm::EventSetup const& eSetup);

private:
	edm::EDGetTokenT<edm::TriggerResults> triggerResults_;
	edm::EDGetTokenT<reco::Centrality> theCentrality_;
	edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster> > theSiPixelCluster;

	std::string triggerPath_;

	int nClusters;
	int minClusters;
	int maxClusters;
	int nEt;
	double minEt;
	double maxEt;

	// histo
	MonitorElement* h_SumEt;
	MonitorElement* h_SiPixelClusters;
	MonitorElement* h_SumEt_SiPixelClusters;
};

#endif
