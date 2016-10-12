#include "DQMOffline/Trigger/interface/BTVHLTOfflineSource.h"

#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "DQMServices/Core/interface/MonitorElement.h"

#include "CommonTools/UtilAlgos/interface/DeltaR.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "TMath.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TPRegexp.h"

#include <cmath>

using namespace edm;
using namespace reco;
using namespace std;
using namespace trigger;

BTVHLTOfflineSource::BTVHLTOfflineSource(const edm::ParameterSet& iConfig)
{
	LogDebug("BTVHLTOfflineSource") << "constructor....";

	dirname_                = iConfig.getUntrackedParameter("dirname",std::string("HLT/BTV/"));
	processname_            = iConfig.getParameter<std::string>("processname");
	verbose_                = iConfig.getUntrackedParameter< bool >("verbose", false);
	triggerSummaryLabel_    = iConfig.getParameter<edm::InputTag>("triggerSummaryLabel");
	triggerResultsLabel_    = iConfig.getParameter<edm::InputTag>("triggerResultsLabel");
	triggerSummaryToken     = consumes <trigger::TriggerEvent> (triggerSummaryLabel_);
	triggerResultsToken     = consumes <edm::TriggerResults>   (triggerResultsLabel_);
	triggerSummaryFUToken   = consumes <trigger::TriggerEvent> (edm::InputTag(triggerSummaryLabel_.label(),triggerSummaryLabel_.instance(),std::string("FU")));
	triggerResultsFUToken   = consumes <edm::TriggerResults>   (edm::InputTag(triggerResultsLabel_.label(),triggerResultsLabel_.instance(),std::string("FU")));
	csvCaloTagsToken_       = consumes<reco::JetTagCollection> (edm::InputTag("hltCombinedSecondaryVertexBJetTagsCalo"));
	csvPfTagsToken_         = consumes<reco::JetTagCollection> (edm::InputTag("hltCombinedSecondaryVertexBJetTagsPF"));
	offlineCSVTokenPF_      = consumes<reco::JetTagCollection> (iConfig.getParameter<edm::InputTag>("offlineCSVLabelPF"));
	offlineCSVTokenCalo_    = consumes<reco::JetTagCollection> (iConfig.getParameter<edm::InputTag>("offlineCSVLabelCalo"));
	hltFastPVToken_         = consumes<std::vector<reco::Vertex> > (iConfig.getParameter<edm::InputTag>("hltFastPVLabel"));
	hltPFPVToken_           = consumes<std::vector<reco::Vertex> > (iConfig.getParameter<edm::InputTag>("hltPFPVLabel"));
	hltCaloPVToken_         = consumes<std::vector<reco::Vertex> > (iConfig.getParameter<edm::InputTag>("hltCaloPVLabel"));
	offlinePVToken_         = consumes<std::vector<reco::Vertex> > (iConfig.getParameter<edm::InputTag>("offlinePVLabel"));

        triggerFilterPFbfCSV_   = iConfig.getParameter<edm::InputTag>("triggerFilterPFbfCSV");
        triggerFilterPFafCSV_   = iConfig.getParameter<edm::InputTag>("triggerFilterPFafCSV");        

        triggerPathPF_          = iConfig.getParameter<std::string>("triggerPathPF");

	std::vector<edm::ParameterSet> paths =  iConfig.getParameter<std::vector<edm::ParameterSet> >("pathPairs");
	for(std::vector<edm::ParameterSet>::iterator pathconf = paths.begin() ; pathconf != paths.end();  pathconf++) { 
		custompathnamepairs_.push_back(make_pair(
					pathconf->getParameter<std::string>("pathname"),
					pathconf->getParameter<std::string>("pathtype")
					));}
}

BTVHLTOfflineSource::~BTVHLTOfflineSource()
{ 
}

void BTVHLTOfflineSource::dqmBeginRun(const edm::Run& run, const edm::EventSetup& c)
{
	bool changed(true);
	if (!hltConfig_.init(run, c, processname_, changed)) {
		LogDebug("BTVHLTOfflineSource") << "HLTConfigProvider failed to initialize.";
	}

	const unsigned int numberOfPaths(hltConfig_.size());
	for(unsigned int i=0; i!=numberOfPaths; ++i){
		pathname_      = hltConfig_.triggerName(i);
		filtername_    = "dummy";
		unsigned int usedPrescale = 1;
		unsigned int objectType = 0;
		std::string triggerType = "";
		bool trigSelected = false;

		for (std::vector<std::pair<std::string, std::string> >::iterator custompathnamepair = custompathnamepairs_.begin(); 
				custompathnamepair != custompathnamepairs_.end(); ++custompathnamepair){
			if(pathname_.find(custompathnamepair->first)!=std::string::npos) { trigSelected = true; triggerType = custompathnamepair->second;}
		}

		if (!trigSelected) continue;

		hltPathsAll_.push_back(PathInfo(usedPrescale, pathname_, "dummy", processname_, objectType, triggerType)); 
	}


}

	void
BTVHLTOfflineSource::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{ 
	iEvent.getByToken(triggerResultsToken, triggerResults_);
	if(!triggerResults_.isValid()) {
		iEvent.getByToken(triggerResultsFUToken,triggerResults_);
		if(!triggerResults_.isValid()) {
			edm::LogInfo("BTVHLTOfflineSource") << "TriggerResults not found, "
				"skipping event";
			return;
		}
	}

	triggerNames_ = iEvent.triggerNames(*triggerResults_);

	iEvent.getByToken(triggerSummaryToken,triggerObj_);
	if(!triggerObj_.isValid()) {
		iEvent.getByToken(triggerSummaryFUToken,triggerObj_);
		if(!triggerObj_.isValid()) {
			edm::LogInfo("BTVHLTOfflineSource") << "TriggerEvent not found, "
				"skipping event";
			return;
		}
	} 

	iEvent.getByToken(csvCaloTagsToken_, csvCaloTags);
	iEvent.getByToken(csvPfTagsToken_, csvPfTags);

	Handle<reco::VertexCollection> VertexHandler;

	Handle<reco::JetTagCollection> offlineJetTagHandlerPF;
	iEvent.getByToken(offlineCSVTokenPF_, offlineJetTagHandlerPF);

	Handle<reco::JetTagCollection> offlineJetTagHandlerCalo;
	iEvent.getByToken(offlineCSVTokenCalo_, offlineJetTagHandlerCalo);

	Handle<reco::VertexCollection> offlineVertexHandler;
	iEvent.getByToken(offlinePVToken_, offlineVertexHandler);

	if(verbose_ && iEvent.id().event()%10000==0)
		cout<<"Run = "<<iEvent.id().run()<<", LS = "<<iEvent.luminosityBlock()<<", Event = "<<iEvent.id().event()<<endl;  

	if(!triggerResults_.isValid()) return;

	for(PathInfoCollection::iterator v = hltPathsAll_.begin(); v!= hltPathsAll_.end(); ++v ){
		unsigned index = triggerNames_.triggerIndex(v->getPath()); 
		if (index < triggerNames_.size() ){     
			
                        // PARTICLE FLOW PATH
                        float DR  = 9999.;
			if (csvPfTags.isValid() && v->getTriggerType() == "PF")
			{
				auto iter = csvPfTags->begin();

				float CSV_online = iter->second;
				if (CSV_online<0) CSV_online = -0.05;

				v->getMEhisto_CSV()->Fill(CSV_online);  
				v->getMEhisto_Pt()->Fill(iter->first->pt()); 
				v->getMEhisto_Eta()->Fill(iter->first->eta());

				DR  = 9999.;
				if(offlineJetTagHandlerPF.isValid()){
					for ( reco::JetTagCollection::const_iterator iterO = offlineJetTagHandlerPF->begin(); iterO != offlineJetTagHandlerPF->end(); iterO++ ){ 
						float CSV_offline = iterO->second;
						if (CSV_offline<0) CSV_offline = -0.05;
						DR = reco::deltaR(iterO->first->eta(),iterO->first->phi(),iter->first->eta(),iter->first->phi());
						if (DR<0.3) {
							v->getMEhisto_CSV_RECOvsHLT()->Fill(CSV_offline,CSV_online); 
							v->getMEhisto_CSV_HLTMinusRECO()->Fill(CSV_online-CSV_offline);
                                                        v->getMEhisto_CSV_RECO()->Fill(CSV_offline);

                                                        //std::cout << "triggerPathPF_:        " << triggerPathPF_        << std::endl;
                                                        //std::cout << "triggerFiltePFbfCSV_:  " << triggerFilterPFbfCSV_ << std::endl;
                                                        //std::cout << "triggerFilterPFafCSV_: " << triggerFilterPFafCSV_ << std::endl;

							//std::cout << "size filters : " << triggerObj_->sizeFilters() << std::endl;
                                                        //for(int it = 0; it < triggerObj_->sizeFilters(); it++)
                                                        //{   
							//	std::cout << triggerObj_->filterLabel(it) << std::endl;
                                                        //}
							
                                                        size_t filterIndex = triggerObj_->filterIndex( triggerFilterPFbfCSV_ );
							if( !(filterIndex >= triggerObj_->sizeFilters()) )
							{
                                                                v->getMEhisto_CSVbeforefilter_RECO()->Fill(CSV_offline);
							}
                                                        
							filterIndex = triggerObj_->filterIndex( triggerFilterPFafCSV_ );
                                                        if( !(filterIndex >= triggerObj_->sizeFilters()) )
                                                        {
                                                                v->getMEhisto_CSVafterfilter_RECO()->Fill(CSV_offline);
                                                        }
								
							continue;
						}
					}
				}

				iEvent.getByToken(hltPFPVToken_, VertexHandler);
				if (VertexHandler.isValid())
				{ 
					v->getMEhisto_PVz()->Fill(VertexHandler->begin()->z()); 
					if (offlineVertexHandler.isValid()) v->getMEhisto_PVz_HLTMinusRECO()->Fill(VertexHandler->begin()->z()-offlineVertexHandler->begin()->z());
				}
			}

                        // Produce turn-on...
                        for (int i = 1; i < 55; i++)

                        {
                            float ratio = (v->getMEhisto_CSV_RECO()->getBinContent(i) != 0) ? v->getMEhisto_CSVbeforefilter_RECO()->getBinContent(i) / v->getMEhisto_CSV_RECO()->getBinContent(i) : 0 ;
                            std::cout << "ratio 1: " << ratio << std::endl;
                            v->getMEhisto_CSVratio_calofilter()->setBinContent(i, ratio);
                        }


                        for (int i = 1; i < 55; i++)
                        {
                            float ratio = (v->getMEhisto_CSV_RECO()->getBinContent(i) != 0) ? v->getMEhisto_CSVafterfilter_RECO()->getBinContent(i) / v->getMEhisto_CSV_RECO()->getBinContent(i) : 0 ;
                            std::cout << "ratio 2: " << ratio << std::endl;
                            v->getMEhisto_CSVratio_PFfilter()->setBinContent(i, ratio);
                        }

                        // CALO PATH
			if (csvCaloTags.isValid() && v->getTriggerType() == "Calo" && !csvCaloTags->empty()) 
			{ 
				auto iter = csvCaloTags->begin();

				float CSV_online = iter->second;
				if (CSV_online<0) CSV_online = -0.05;

				v->getMEhisto_CSV()->Fill(CSV_online);  
				v->getMEhisto_Pt()->Fill(iter->first->pt()); 
				v->getMEhisto_Eta()->Fill(iter->first->eta());

				DR  = 9999.;
				if(offlineJetTagHandlerCalo.isValid()){
					for ( reco::JetTagCollection::const_iterator iterO = offlineJetTagHandlerCalo->begin(); iterO != offlineJetTagHandlerCalo->end(); iterO++ )
					{
						float CSV_offline = iterO->second;
						if (CSV_offline<0) CSV_offline = -0.05;
						DR = reco::deltaR(iterO->first->eta(),iterO->first->phi(),iter->first->eta(),iter->first->phi());
						if (DR<0.3) 
						{
							v->getMEhisto_CSV_RECOvsHLT()->Fill(CSV_offline,CSV_online);
							v->getMEhisto_CSV_HLTMinusRECO()->Fill(CSV_online-CSV_offline);
							v->getMEhisto_CSV_RECO()->Fill(CSV_offline);
                                                        v->getMEhisto_CSVafterfilter_RECO()->Fill(CSV_offline);
							continue;
						} 
					}     
				}

				iEvent.getByToken(hltFastPVToken_, VertexHandler);
				if (VertexHandler.isValid()) 
				{
					v->getMEhisto_PVz()->Fill(VertexHandler->begin()->z()); 
					if (offlineVertexHandler.isValid()) v->getMEhisto_fastPVz_HLTMinusRECO()->Fill(VertexHandler->begin()->z()-offlineVertexHandler->begin()->z());
				}

				iEvent.getByToken(hltCaloPVToken_, VertexHandler);
				if (VertexHandler.isValid())
				{
					v->getMEhisto_fastPVz()->Fill(VertexHandler->begin()->z()); 
					if (offlineVertexHandler.isValid()) v->getMEhisto_PVz_HLTMinusRECO()->Fill(VertexHandler->begin()->z()-offlineVertexHandler->begin()->z());
				}
			}

                        // Produce turn-on...
                        for (int i = 1; i < 55; i++)

                        {   
                            float ratio = (v->getMEhisto_CSV_RECO()->getBinContent(i) != 0) ? v->getMEhisto_CSVbeforefilter_RECO()->getBinContent(i) / v->getMEhisto_CSV_RECO()->getBinContent(i) : 0 ;
                            std::cout << "ratio 1: " << ratio << std::endl;
                            v->getMEhisto_CSVratio_calofilter()->setBinContent(i, ratio);
                        }


                        for (int i = 1; i < 55; i++)
                        {   
                            float ratio = (v->getMEhisto_CSV_RECO()->getBinContent(i) != 0) ? v->getMEhisto_CSVafterfilter_RECO()->getBinContent(i) / v->getMEhisto_CSV_RECO()->getBinContent(i) : 0 ;
                            std::cout << "ratio 2: " << ratio << std::endl;
                            v->getMEhisto_CSVratio_PFfilter()->setBinContent(i, ratio);
                        }
		}
	}
}

	void 
BTVHLTOfflineSource::bookHistograms(DQMStore::IBooker & iBooker, edm::Run const & run, edm::EventSetup const & c)
{
	iBooker.setCurrentFolder(dirname_);
	for(PathInfoCollection::iterator v = hltPathsAll_.begin(); v!= hltPathsAll_.end(); ++v ){
		//
		std::string trgPathName = HLTConfigProvider::removeVersion(v->getPath());
		std::string subdirName  = dirname_ +"/"+ trgPathName;
		std::string trigPath    = "("+trgPathName+")";
		iBooker.setCurrentFolder(subdirName);  

		std::string labelname("HLT");
		std::string histoname(labelname+"");
		std::string title(labelname+"");

		histoname = labelname+"_CSV";
		title = labelname+"_CSV "+trigPath;
		MonitorElement * CSV =  iBooker.book1D(histoname.c_str(),title.c_str(),110,-0.1,1);

		histoname = labelname+"_Pt";
		title = labelname+"_Pt "+trigPath;
		MonitorElement * Pt =  iBooker.book1D(histoname.c_str(),title.c_str(),100,0,400);

		histoname = labelname+"_Eta";
		title = labelname+"_Eta "+trigPath;
		MonitorElement * Eta =  iBooker.book1D(histoname.c_str(),title.c_str(),60,-3.0,3.0);

		histoname = "RECO_CSV";
		title = "offline CSV "+trigPath;
		MonitorElement * CSV_RECO = iBooker.book1D(histoname.c_str(),title.c_str(),55,-0.1,1);

		histoname = "HLTMinusRECO_CSV";
		title = "online CSV - offline CSV "+trigPath;
		MonitorElement * CSV_HLTMinusRECO = iBooker.book1D(histoname.c_str(),title.c_str(),100,-2,2);

		histoname = "RECOvsHLT_CSV";
		title = "offline CSV vs online CSV "+trigPath;
		MonitorElement * CSV_RECOvsHLT =  iBooker.book2D(histoname.c_str(),title.c_str(),110,-0.1,1,110,-0.1,1);

                histoname = "RECO_CSVbeforefilter";
                title = "offline before after filter "+trigPath;
                MonitorElement * CSVbeforefilter_RECO = iBooker.book1D(histoname.c_str(),title.c_str(),55,-0.1,1);

		histoname = "RECO_CSVafterfilter";
		title = "offline CSV after filter "+trigPath;
		MonitorElement * CSVafterfilter_RECO = iBooker.book1D(histoname.c_str(),title.c_str(),55,-0.1,1);

                histoname = "CSVTurnON";
                title = "turn on CSV "+trigPath;
                MonitorElement * CSV_turnon = iBooker.book1D(histoname.c_str(),title.c_str(),110,-0.1,1);

                histoname = "caloTurnON";
                title = "ratio of offline CSV before and after calo b-tag filter"+trigPath;
                MonitorElement * CSVratio_calofilter = iBooker.book1D(histoname.c_str(),title.c_str(),55,-0.1,1);

                histoname = "PFTurnON";
                title = "ratio of offline CSV before and after PF b-tag filter"+trigPath;
                MonitorElement * CSVratio_PFfilter = iBooker.book1D(histoname.c_str(),title.c_str(),55,-0.1,1);

		histoname = labelname+"_PVz";
		title = "online z(PV) "+trigPath;
		MonitorElement * PVz =  iBooker.book1D(histoname.c_str(),title.c_str(),80,-20,20);

		histoname = labelname+"_fastPVz";
		title = "online z(fastPV) "+trigPath;
		MonitorElement * fastPVz =  iBooker.book1D(histoname.c_str(),title.c_str(),80,-20,20);

		histoname = "HLTMinusRECO_PVz";
		title = "online z(PV) - offline z(PV) "+trigPath;
		MonitorElement * PVz_HLTMinusRECO =  iBooker.book1D(histoname.c_str(),title.c_str(),200,-0.5,0.5);

		histoname = "HLTMinusRECO_fastPVz";
		title = "online z(fastPV) - offline z(PV) "+trigPath;
		MonitorElement * fastPVz_HLTMinusRECO =  iBooker.book1D(histoname.c_str(),title.c_str(),100,-2,2);

		v->setHistos(CSV,Pt,Eta,CSV_RECO,CSV_HLTMinusRECO,CSV_RECOvsHLT,CSVbeforefilter_RECO,CSVafterfilter_RECO,CSV_turnon,CSVratio_calofilter,CSVratio_PFfilter,PVz,fastPVz,PVz_HLTMinusRECO,fastPVz_HLTMinusRECO);
	}
}
