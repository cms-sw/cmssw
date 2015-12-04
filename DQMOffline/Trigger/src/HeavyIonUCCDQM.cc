#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DQMOffline/Trigger/interface/HeavyIonUCCDQM.h"

HeavyIonUCCDQM::HeavyIonUCCDQM(const edm::ParameterSet& ps)
{
	triggerResults_ = consumes<edm::TriggerResults>(ps.getParameter<edm::InputTag>("triggerResults"));
	theCentrality_ = consumes<reco::Centrality>(ps.getParameter<edm::InputTag>("centralityTag"));
	theSiPixelCluster = consumes<edmNew::DetSetVector<SiPixelCluster> >(ps.getParameter<edm::InputTag>("pixelCluster"));
	triggerPath_ = ps.getParameter<std::string>("triggerPath");

	nClusters = ps.getParameter<int>("nClusters");
	minClusters = ps.getParameter<int>("minClusters");
	maxClusters = ps.getParameter<int>("maxClusters");
	nEt = ps.getParameter<int>("nEt");
	minEt = ps.getParameter<double>("minEt");
	maxEt = ps.getParameter<double>("maxEt");
}

HeavyIonUCCDQM::~HeavyIonUCCDQM()
{

}


void HeavyIonUCCDQM::bookHistograms(DQMStore::IBooker & ibooker_, edm::Run const &, edm::EventSetup const &)
{
	ibooker_.cd();;
	ibooker_.setCurrentFolder("HLT/HI/" + triggerPath_);

	h_SumEt = ibooker_.book1D("h_SumEt", "SumEt", nEt, minEt, maxEt);
	h_SiPixelClusters = ibooker_.book1D("h_SiPixelClusters", "h_SiPixelClusters", nClusters, minClusters, maxClusters);
	h_SumEt_SiPixelClusters = ibooker_.book2D("h_SumEt_SiPixelClusters","h_SumEt_SiPixelClusters",nEt, minEt, maxEt, nClusters, minClusters, maxClusters);

	ibooker_.cd();
}

void HeavyIonUCCDQM::analyze(edm::Event const& e, edm::EventSetup const& eSetup)
{

	edm::Handle<edm::TriggerResults> hltresults;
	e.getByToken(triggerResults_,hltresults);
	if(!hltresults.isValid())
	{
		edm::LogError ("HeavyIonUCCDQM") << "invalid collection: TriggerResults" << "\n";
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

	if (!hasFired) return;

	edm::Handle<edmNew::DetSetVector<SiPixelCluster> > cluster;
	e.getByToken(theSiPixelCluster, cluster);
	if ( cluster.isValid() ) {
		h_SiPixelClusters->Fill(cluster->dataSize());
	}

	edm::Handle<reco::Centrality> hiCentrality;
	e.getByToken(theCentrality_, hiCentrality);
	if ( hiCentrality.isValid() ) {
		h_SumEt->Fill( hiCentrality->EtHFtowerSum());
	}

	if ( cluster.isValid() && hiCentrality.isValid() ) {
	        h_SumEt_SiPixelClusters->Fill( hiCentrality->EtHFtowerSum(), cluster->dataSize());
	}


}

DEFINE_FWK_MODULE(HeavyIonUCCDQM);
