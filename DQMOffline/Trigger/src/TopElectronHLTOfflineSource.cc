#include "DQMOffline/Trigger/interface/TopElectronHLTOfflineSource.h"

#include "DQMServices/Core/interface/MonitorElement.h"

#include "FWCore/Framework/interface/Run.h"

#include <boost/algorithm/string.hpp>

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "DQMOffline/Trigger/interface/TopElectronHLTOfflineSource.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

//using namespace egHLT;

TopElectronHLTOfflineSource::TopElectronHLTOfflineSource(const edm::ParameterSet& conf) :
  beamSpot_(consumes<reco::BeamSpot>(conf.getParameter<edm::InputTag>("beamSpot"))) {

	dbe_ = edm::Service<DQMStore>().operator->();
	
	if (!dbe_) 
	{
		edm::LogInfo("TopElectronHLTOfflineSource") << "unable to get DQMStore service?";
	}
	
	if(conf.getUntrackedParameter<bool>("DQMStore", false)) 
	{
	  if(!dbe_) dbe_->setVerbose(0);
	}

	dirName_ = conf.getParameter<std::string>("DQMDirName");
	
	electronIdNames_ = conf.getParameter<std::vector<std::string> >("electronIdNames");
	hltTag_ = conf.getParameter<std::string>("hltTag");
	superTriggerNames_ = conf.getParameter<std::vector<std::string> >("superTriggerNames");
	electronTriggerNames_ = conf.getParameter<std::vector<std::string> >("electronTriggerNames");
	
	triggerResultsLabel_ = consumes<edm::TriggerResults>(conf.getParameter<edm::InputTag>("triggerResultsLabel"));
	triggerSummaryLabel_ = consumes<trigger::TriggerEvent>(conf.getParameter<edm::InputTag>("triggerSummaryLabel"));
	electronLabel_ = consumes<reco::GsfElectronCollection>(conf.getParameter<edm::InputTag>("electronCollection"));
	primaryVertexLabel_ = consumes<reco::VertexCollection>(conf.getParameter<edm::InputTag>("primaryVertexCollection"));
	triggerJetFilterLabel_	= conf.getParameter<edm::InputTag>("triggerJetFilterLabel");
	triggerElectronFilterLabel_ = conf.getParameter<edm::InputTag>("triggerElectronFilterLabel");

	excludeCloseJets_ = conf.getParameter<bool>("excludeCloseJets");
	requireTriggerMatch_ = conf.getParameter<bool>("requireTriggerMatch");
	electronMinEt_ = conf.getParameter<double>("electronMinEt");
	electronMaxEta_ = conf.getParameter<double>("electronMaxEta");
		
	addExtraId_ = conf.getParameter<bool>("addExtraId");
	extraIdCutsSigmaEta_ = conf.getParameter<double>("extraIdCutsSigmaEta");
	extraIdCutsSigmaPhi_ = conf.getParameter<double>("extraIdCutsSigmaPhi");
	extraIdCutsDzPV_ = conf.getParameter<double>("extraIdCutsDzPV");

}
TopElectronHLTOfflineSource::~TopElectronHLTOfflineSource()
{
}

void TopElectronHLTOfflineSource::beginJob()
{
  if(!dbe_) return;
	dbe_->setCurrentFolder(dirName_);
	for (size_t i = 0; i < superTriggerNames_.size(); ++i)
	{
		eleMEs_.push_back(EleMEs(dbe_, electronIdNames_, addExtraId_, superTriggerNames_[i]));
		for (size_t j = 0; j < electronTriggerNames_.size(); ++j)
		{
			eleMEs_.push_back(EleMEs(dbe_, electronIdNames_, addExtraId_, superTriggerNames_[i]+"_"+electronTriggerNames_[j]));
			//std::cout <<superTriggerNames_[i]+"_"+electronTriggerNames_[j]<<std::endl;
			
		}
	}
  for (size_t i = 0; i < electronIdNames_.size(); i++)
    eleIdTokenCollection_.push_back(consumes<edm::ValueMap<float> >((edm::InputTag)electronIdNames_[i])); 
	//std::cout <<"done"<<std::endl;
}
void TopElectronHLTOfflineSource::setupHistos(std::vector<EleMEs> topEleHists)
{
	for (size_t i = 0; i < eleMEs_.size(); ++i)
	{
		topEleHists.push_back(eleMEs_[i]);
	}
}
void TopElectronHLTOfflineSource::endJob()
{
}
void TopElectronHLTOfflineSource::beginRun(const edm::Run& run, const edm::EventSetup& c)
{
  hltConfigValid_=hltConfig_.init(run,c,hltTag_,hltConfigChanged_);
}
void TopElectronHLTOfflineSource::endRun(const edm::Run& run, const edm::EventSetup& c)
{
}

void TopElectronHLTOfflineSource::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
        if(!dbe_) return;
	// ---- Get Trigger Decisions for all triggers under investigation ----
	edm::Handle<edm::TriggerResults> hltResults;
	if(!iEvent.getByToken(triggerResultsLabel_, hltResults) || !hltResults.product()) return; //bail if we didnt get trigger results
      

	
	if (!hltConfigValid_) return;
	
	std::vector<bool> superTriggerAccepts;
	std::vector<bool> electronTriggerAccepts;
	
	for (size_t i = 0; i < superTriggerNames_.size(); ++i)
	{
		unsigned int triggerIndex( hltConfig_.triggerIndex(superTriggerNames_[i]) );
		bool accept = false;
		
		if (triggerIndex < hltResults->size())
		{
			accept = hltResults->accept(triggerIndex);
		}
		
		superTriggerAccepts.push_back(accept);
	}
	
	for (size_t i = 0; i < electronTriggerNames_.size(); ++i)
	{
		unsigned int triggerIndex( hltConfig_.triggerIndex(electronTriggerNames_[i]) );
		bool accept = false;
		
		if (triggerIndex < hltResults->size())
		{
			accept = hltResults->accept(triggerIndex);
		} 
		
		electronTriggerAccepts.push_back(accept);
	}
	
	// get reconstructed electron collection
	if(!iEvent.getByToken(electronLabel_, eleHandle_) || !eleHandle_.product()) return;
	
	// Get Trigger Event, providing the information about trigger objects	
	if(!iEvent.getByToken(triggerSummaryLabel_, triggerEvent_) || !triggerEvent_.product()) return; 
	
	edm::Handle<reco::VertexCollection> vertexHandle;
        if(!iEvent.getByToken(primaryVertexLabel_, vertexHandle) || !vertexHandle.product()) return;

        reco::Vertex::Point vertexPoint(0., 0., 0.);
        if (vertexHandle.product()->size() != 0)
        {
                const reco::Vertex& theVertex = vertexHandle.product()->front();
                vertexPoint = theVertex.position();
        }
	else
	{
		edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
		if(!iEvent.getByToken(beamSpot_, recoBeamSpotHandle) ||  !recoBeamSpotHandle.product()) return;

		vertexPoint = recoBeamSpotHandle->position();
	}
	
	trigger::size_type jetFilterPos = triggerEvent_->filterIndex(triggerJetFilterLabel_);
	std::vector<const trigger::TriggerObject*> triggerJets;
	
	if (jetFilterPos != triggerEvent_->sizeFilters())
	{
		for (size_t i = 0; i < triggerEvent_->filterKeys(jetFilterPos).size(); ++i)
		{
			size_t objNr = triggerEvent_->filterKeys(jetFilterPos)[i];
			if(objNr<triggerEvent_->sizeObjects()){
			  triggerJets.push_back(& triggerEvent_->getObjects()[objNr]);
			}
		}	
	}
	
	trigger::size_type eleFilterPos = triggerEvent_->filterIndex(triggerElectronFilterLabel_);
	std::vector<const trigger::TriggerObject*> triggerElectrons;
 
	if (eleFilterPos != triggerEvent_->sizeFilters())
	{
		for (size_t i = 0; i < triggerEvent_->filterKeys(eleFilterPos).size(); ++i)
		{
			size_t objNr = triggerEvent_->filterKeys(eleFilterPos)[i];
			if(objNr<triggerEvent_->sizeObjects()){
			  triggerElectrons.push_back(& triggerEvent_->getObjects()[objNr]);
			}
		}
	}
	
	const reco::GsfElectronCollection& eles = *eleHandle_;
	
	for(size_t eleNr=0; eleNr < eles.size(); ++eleNr)
	{
		
		const reco::GsfElectron& ele = eles[eleNr];
		
		// electron selection
		
		if(ele.et() > electronMinEt_ && std::abs(ele.eta()) < electronMaxEta_)
		{
			size_t index = 0;
			for (size_t i = 0; i < superTriggerNames_.size(); ++i)
			{
				if (superTriggerAccepts[i])
					fill(eleMEs_[index], iEvent, eleNr, triggerJets, triggerElectrons, vertexPoint);
				index++;
				
				for (size_t j = 0; j < electronTriggerNames_.size(); ++j)
				{
					if (superTriggerAccepts[i] && electronTriggerAccepts[j]) 
						fill(eleMEs_[index], iEvent, eleNr, triggerJets, triggerElectrons, vertexPoint);
					index++;
				}
			}
		}
	}
}

void TopElectronHLTOfflineSource::EleMEs::setup(DQMStore* dbe, const std::vector<std::string>& eleIdNames, bool addExtraId, const std::string& name)
{ 
	for (size_t i = 0; i < eleIdNames.size(); ++i)
	{
		eleIdNames_.push_back(eleIdNames[i]);
		if (addExtraId)
			eleIdNames_.push_back(eleIdNames[i]+"extraId");
	}
	
	addMESets(name);
	
	for (size_t i = 0; i < eleMESets_.size(); ++i)
	{
		setupMESet(eleMESets_[i], dbe, name+"_"+fullName(i));
		LogDebug("TopElectronHLTOfflineSource") << "Booked MonitorElement with name " << name;
	}
}
void TopElectronHLTOfflineSource::EleMEs::setupMESet(EleMESet& eleSet, DQMStore* dbe, const std::string& name)
{
	eleSet.ele_et = dbe->book1D("ele_"+name+"_et", "ele_"+name+"_et", 50, 0., 500.);
	eleSet.ele_eta = dbe->book1D("ele_"+name+"_eta", "ele_"+name+"_eta", 50, -2.5, 2.5);
	eleSet.ele_phi = dbe->book1D("ele_"+name+"_phi","ele_"+name+"_phi", 50, -3.1416, 3.1416);
	eleSet.ele_isolEm = dbe->book1D("ele_"+name+"_isolEm", "ele_"+name+"_isolEm", 50, -0.05, 3.);
	eleSet.ele_isolHad = dbe->book1D("ele_"+name+"_isolHad", "ele_"+name+"_isolHad", 50, -0.05, 5.);
	eleSet.ele_minDeltaR = dbe->book1D("ele_"+name+"_minDeltaR", "ele_"+name+"_minDeltaR", 50, 0., 1.);
	eleSet.global_n30jets = dbe->book1D("ele_"+name+"_global_n30jets", "ele_"+name+"_global_n30jets", 10, -0.5, 9.5);
	eleSet.global_sumEt = dbe->book1D("ele_"+name+"_global_sumEt", "ele_"+name+"_global_sumEt", 50, 0., 1000.);
	eleSet.ele_gsftrack_etaError = dbe->book1D("ele_"+name+"_gsftrack_etaError", "ele_"+name+"_gsftrack_etaError", 50, 0., 0.005);
	eleSet.ele_gsftrack_phiError = dbe->book1D("ele_"+name+"_gsftrack_phiError", "ele_"+name+"_gsftrack_phiError", 50, 0., 0.005);
	eleSet.ele_gsftrack_numberOfValidHits = dbe->book1D("ele_"+name+"_gsftrack_numberOfValidHits", "ele_"+name+"_gsftrack_numberOfValidHits", 25, -0.5, 24.5);
	eleSet.ele_gsftrack_dzPV = dbe->book1D("ele_"+name+"_gsftrack_dzPV", "ele_"+name+"_gsftrack_dzPV", 50, 0., 0.2);
}

void TopElectronHLTOfflineSource::EleMEs::addMESets(const std::string& name)
{
	eleMENames_.push_back("EB");
	eleMENames_.push_back("EE");
	name_ = name;
	for (size_t i=0; i < eleIdNames_.size() * eleMENames_.size(); ++i)
	{
		eleMESets_.push_back(EleMESet());
	}
}

void TopElectronHLTOfflineSource::fill(EleMEs& eleMEs, const edm::Event& iEvent, size_t eleIndex, const std::vector<const trigger::TriggerObject*>& triggerJets, const std::vector<const trigger::TriggerObject*>& triggerElectrons, const reco::Vertex::Point& vertexPoint)
{
	const reco::GsfElectron& ele = (*eleHandle_)[eleIndex];
	
	float dzPV = std::abs(ele.gsfTrack()->dz(vertexPoint));
	
	bool isTriggerMatched = false;
	for (size_t i = 0; i < triggerElectrons.size(); ++i)
	{
		if (deltaR(*(triggerElectrons[i]), ele.p4()) < 0.3)
			 isTriggerMatched = true;
	}
	
	if (requireTriggerMatch_ && !isTriggerMatched)
		return;
	
	// Calculate minimum deltaR to closest jet and sumEt (all jets)
	float minDeltaR = 999.;
	float sumEt = 0.;
	
	for (size_t jetNr = 0; jetNr < triggerJets.size(); ++jetNr)
	{
		const trigger::TriggerObject& jet = *triggerJets[jetNr];

		sumEt += jet.et();
		
		float dr = deltaR(jet, ele.p4());
		
		if (!excludeCloseJets_ && dr < minDeltaR)
			minDeltaR = dr;
		if (excludeCloseJets_ && dr > 0.1 && dr < minDeltaR)
			minDeltaR = dr;
	}
	
	for (size_t j = 0; j < eleMEs.eleIdNames().size(); ++j)
	{
		bool eId = true;

		edm::Handle<edm::ValueMap<float> > eIdMapHandle;
    iEvent.getByToken(eleIdTokenCollection_[j], eIdMapHandle);
		const edm::ValueMap<float>& eIdMap = *eIdMapHandle;
		eId = eIdMap[edm::Ref<reco::GsfElectronCollection>(eleHandle_, eleIndex)];
		
		bool extraId = true;
		if (addExtraId_)
		{
			if (ele.gsfTrack()->etaError() > extraIdCutsSigmaEta_)
				extraId = false;
			if (ele.gsfTrack()->phiError() > extraIdCutsSigmaPhi_)
				extraId = false;
			if (dzPV > extraIdCutsDzPV_)
				extraId = false;
		}
		
		for (size_t i = 0; i < eleMEs.eleMENames().size(); ++i)
		{
			if (eId && eleMEs.eleMENames()[i] == "EB" && ele.isEB()&& !ele.isGap())
				eleMEs.fill(eleMEs.getMESet(i, j), ele, minDeltaR, sumEt, triggerJets.size(), dzPV);
			if (eId && eleMEs.eleMENames()[i] == "EE" && ele.isEE()&& !ele.isGap())
				eleMEs.fill(eleMEs.getMESet(i, j), ele, minDeltaR, sumEt, triggerJets.size(), dzPV);
			if (addExtraId_)
			{
				if (eId && extraId && eleMEs.eleMENames()[i] == "EB" && ele.isEB()&& !ele.isGap())
					eleMEs.fill(eleMEs.getMESet(i, j+1), ele, minDeltaR, sumEt, triggerJets.size(), dzPV);
				if (eId && extraId && eleMEs.eleMENames()[i] == "EE" && ele.isEE()&& !ele.isGap())
					eleMEs.fill(eleMEs.getMESet(i, j+1), ele, minDeltaR, sumEt, triggerJets.size(), dzPV);
			}
		}
		if (addExtraId_)
			++j;

	}
}
void TopElectronHLTOfflineSource::EleMEs::fill(EleMESet& eleMESet, const reco::GsfElectron& ele, float minDeltaR, float sumEt, int n30jets, float dzPV)
{
	LogDebug("TopElectronHLTOfflineSource") << "filling the histos with " << ele.et();

	eleMESet.ele_et->Fill(ele.et());
	eleMESet.ele_eta->Fill(ele.eta());
	eleMESet.ele_phi->Fill(ele.phi());
	eleMESet.ele_isolEm->Fill(ele.dr03EcalRecHitSumEt());
	eleMESet.ele_isolHad->Fill(ele.dr03HcalTowerSumEt());
	eleMESet.ele_minDeltaR->Fill(minDeltaR);
	eleMESet.global_n30jets->Fill(n30jets);
	eleMESet.global_sumEt->Fill(sumEt);
	eleMESet.ele_gsftrack_etaError->Fill(ele.gsfTrack()->etaError());
	eleMESet.ele_gsftrack_phiError->Fill(ele.gsfTrack()->phiError());
	eleMESet.ele_gsftrack_numberOfValidHits->Fill(ele.gsfTrack()->numberOfValidHits());
	eleMESet.ele_gsftrack_dzPV->Fill(dzPV);		
}
