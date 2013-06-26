#include "DQMOffline/Trigger/interface/TopElectronHLTOfflineClient.h"


#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"


TopElectronHLTOfflineClient::TopElectronHLTOfflineClient(const edm::ParameterSet& iConfig) : dbe_(NULL)
{
	dbe_ = edm::Service<DQMStore>().operator->();
	
	if (!dbe_) 
	{
		edm::LogError("TopElectronHLTOfflineClient") << "unable to get DQMStore service, upshot is no client histograms will be made";
	}
	
	if(iConfig.getUntrackedParameter<bool>("DQMStore", false)) 
	{
		if (dbe_) dbe_->setVerbose(0);
	}
	
 	dirName_=iConfig.getParameter<std::string>("DQMDirName");
	
	if (dbe_)
		dbe_->setCurrentFolder(dirName_);

	hltTag_ = iConfig.getParameter<std::string>("hltTag");
	
	electronIdNames_ = iConfig.getParameter<std::vector<std::string> >("electronIdNames");
	superTriggerNames_ = iConfig.getParameter<std::vector<std::string> >("superTriggerNames");
	electronTriggerNames_ = iConfig.getParameter<std::vector<std::string> >("electronTriggerNames");
	addExtraId_ = iConfig.getParameter<bool>("addExtraId");

	runClientEndLumiBlock_ = iConfig.getParameter<bool>("runClientEndLumiBlock");
	runClientEndRun_ = iConfig.getParameter<bool>("runClientEndRun");
	runClientEndJob_ = iConfig.getParameter<bool>("runClientEndJob");

}


TopElectronHLTOfflineClient::~TopElectronHLTOfflineClient()
{ 
}

void TopElectronHLTOfflineClient::beginJob()
{
	//compose the ME names we need
	
	// Eta regions
	std::vector<std::string> regions;
	regions.push_back("EB");
	regions.push_back("EE");
	
	// Electron IDs, including own extra ID
	std::vector<std::string> eleIdNames;
	for (size_t i = 0; i < electronIdNames_.size(); ++i)
	{
		eleIdNames.push_back(electronIdNames_[i]);
		if (addExtraId_)
			eleIdNames.push_back(electronIdNames_[i]+"extraId");
	}
	
	std::vector<std::string> vars;
	vars.push_back("_et");
	vars.push_back("_eta");
	vars.push_back("_phi");
	vars.push_back("_isolEm");
	vars.push_back("_isolHad");
	vars.push_back("_minDeltaR");
	vars.push_back("_global_n30jets");
	vars.push_back("_global_sumEt");
	vars.push_back("_gsftrack_etaError");
	vars.push_back("_gsftrack_phiError");
	vars.push_back("_gsftrack_numberOfValidHits");
	vars.push_back("_gsftrack_dzPV");
	
	
	for (size_t i = 0; i < eleIdNames.size(); ++i)
		for (size_t j = 0; j < regions.size(); ++j)
			for (size_t k = 0; k < vars.size(); ++k)
				for (size_t l = 0; l < superTriggerNames_.size(); ++l)
				{
					superMeNames_.push_back("ele_"+superTriggerNames_[l]+"_"+regions[j]+"_"+eleIdNames[i]+vars[k] );
					for (size_t m = 0; m < electronTriggerNames_.size(); ++m)
					{
						eleMeNames_.push_back("ele_"+superTriggerNames_[l]+"_"+electronTriggerNames_[m] +"_"+regions[j]+"_"+eleIdNames[i]+vars[k]);
					}
				}

	

}

void TopElectronHLTOfflineClient::endJob() 
{
	if(runClientEndJob_)
		runClient_();
}

void TopElectronHLTOfflineClient::beginRun(const edm::Run& run, const edm::EventSetup& c)
{
}


void TopElectronHLTOfflineClient::endRun(const edm::Run& run, const edm::EventSetup& c)
{
	if(runClientEndRun_)
		runClient_();
}

//dummy analysis function
void TopElectronHLTOfflineClient::analyze(const edm::Event& iEvent,const edm::EventSetup& iSetup)
{
}

void TopElectronHLTOfflineClient::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg,const edm::EventSetup& c)
{ 
	if(runClientEndLumiBlock_)
		runClient_();
}

void TopElectronHLTOfflineClient::runClient_()
{
	if (!dbe_) return; //we dont have the DQMStore so we cant do anything
	dbe_->setCurrentFolder(dirName_);
	
	size_t k = 0;
	for (size_t i = 0; i < superMeNames_.size(); ++i)
	{
		for (size_t j = 0; j < electronTriggerNames_.size(); ++j)
		{
			if (k >= eleMeNames_.size())
				continue;
			createSingleEffHists(superMeNames_[i], eleMeNames_[k], eleMeNames_[k]+"_eff");
			++k;
		}
	}
	superTriggerNames_.size();
	electronTriggerNames_.size();


}

void TopElectronHLTOfflineClient::createSingleEffHists(const std::string& denomName, const std::string& nomName, const std::string& effName)
{ 
	MonitorElement* denom = dbe_->get(dirName_+"/"+denomName);
	
	MonitorElement* nom = dbe_->get(dirName_+"/"+nomName);
	if(nom!=NULL && denom!=NULL)
	{

		makeEffMonElemFromPassAndAll(effName, nom, denom);	 
	}
}

	
MonitorElement* TopElectronHLTOfflineClient::makeEffMonElemFromPassAndAll(const std::string& name, const MonitorElement* pass, const MonitorElement* all)
{
	TH1F* passHist = pass->getTH1F();
	if(passHist->GetSumw2N()==0) 
		passHist->Sumw2();
	TH1F* allHist = all->getTH1F();
	if(allHist->GetSumw2N()==0)
		allHist->Sumw2();
	
	TH1F* effHist = (TH1F*) passHist->Clone(name.c_str());
	effHist->Divide(passHist,allHist,1,1,"B");

	MonitorElement* eff = dbe_->get(dirName_+"/"+name);
	if(eff==NULL)
	{
		eff= dbe_->book1D(name,effHist);
	}
	else
	{ //I was having problems with collating the histograms, hence why I'm just reseting the histogram value
		*eff->getTH1F()=*effHist; 
		delete effHist;
	}
	return eff;
}
