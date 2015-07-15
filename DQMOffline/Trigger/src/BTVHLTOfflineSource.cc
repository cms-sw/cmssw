/*
  BTVHLTOffline DQM code
*/

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
  
  //
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
  offlineCSVLabelPF_      = iConfig.getParameter<edm::InputTag>("offlineCSVLabelPF");
  offlineCSVLabelCalo_    = iConfig.getParameter<edm::InputTag>("offlineCSVLabelCalo");
  hltFastPVLabel_         = iConfig.getParameter<edm::InputTag>("hltFastPVLabel");
  hltPFPVLabel_           = iConfig.getParameter<edm::InputTag>("hltPFPVLabel");
  hltCaloPVLabel_         = iConfig.getParameter<edm::InputTag>("hltCaloPVLabel");
  offlinePVLabel_         = iConfig.getParameter<edm::InputTag>("offlinePVLabel");
 
  //
  std::vector<edm::ParameterSet> paths =  iConfig.getParameter<std::vector<edm::ParameterSet> >("pathPairs");
  for(std::vector<edm::ParameterSet>::iterator pathconf = paths.begin() ; pathconf != paths.end();  pathconf++) { 
    custompathnamepairs_.push_back(make_pair(
					     pathconf->getParameter<std::string>("pathname"),
					     pathconf->getParameter<std::string>("pathtype")
					     ));}
					     

}
//------------------------------------------------------------------------//
BTVHLTOfflineSource::~BTVHLTOfflineSource()
{ 
  //
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//------------------------------------------------------------------------//
void
BTVHLTOfflineSource::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{ 
 
  //---------- triggerResults ----------
  
  iEvent.getByToken(triggerResultsToken, triggerResults_);
  if(!triggerResults_.isValid()) {
    iEvent.getByToken(triggerResultsFUToken,triggerResults_);
    if(!triggerResults_.isValid()) {
      edm::LogInfo("BTVHLTOfflineSource") << "TriggerResults not found, "
	"skipping event";
      return;
    }
  }
  
  //---------- triggerResults ----------
  
  triggerNames_ = iEvent.triggerNames(*triggerResults_);
  
  
  //---------- triggerSummary ----------
  
  iEvent.getByToken(triggerSummaryToken,triggerObj_);
  if(!triggerObj_.isValid()) {
    iEvent.getByToken(triggerSummaryFUToken,triggerObj_);
    if(!triggerObj_.isValid()) {
      edm::LogInfo("BTVHLTOfflineSource") << "TriggerEvent not found, "
	"skipping event";
      return;
    }
  } 
  
  //------------ Online Objects -------
  
  iEvent.getByToken(csvCaloTagsToken_, csvCaloTags);
  iEvent.getByToken(csvPfTagsToken_, csvPfTags);
  
  Handle<reco::VertexCollection> VertexHandler;
  
  //------------ Offline Objects ------
   
  Handle<reco::JetTagCollection> offlineJetTagHandlerPF;
  iEvent.getByLabel(offlineCSVLabelPF_, offlineJetTagHandlerPF);
  
  Handle<reco::JetTagCollection> offlineJetTagHandlerCalo;
  iEvent.getByLabel(offlineCSVLabelCalo_, offlineJetTagHandlerCalo);
  
  Handle<reco::VertexCollection> offlineVertexHandler;
  iEvent.getByLabel(offlinePVLabel_, offlineVertexHandler);
    
  
  //---------- Event counting (DEBUG) ----------
  
  if(verbose_ && iEvent.id().event()%10000==0)
    cout<<"Run = "<<iEvent.id().run()<<", LS = "<<iEvent.luminosityBlock()<<", Event = "<<iEvent.id().event()<<endl;  
  
  if(!triggerResults_.isValid()) return;
  
  
  //---------- Fill histograms ---------------
   
  for(PathInfoCollection::iterator v = hltPathsAll_.begin(); v!= hltPathsAll_.end(); ++v ){
    unsigned index = triggerNames_.triggerIndex(v->getPath()); 
    if (index < triggerNames_.size() ){
     
     float DR  = 9999.;
     //
     if (csvPfTags.isValid() && v->getTriggerType() == "PF")
     {
      auto iter = csvPfTags->begin();
     
      //std::cout <<"PF "<< iter->second <<" " << iter->first->pt() << std::endl;
      
      float CSV_online = iter->second;
      if (CSV_online<0) CSV_online = -0.05;
    
      v->getMEhisto_CSV()->Fill(CSV_online);  
      v->getMEhisto_Pt()->Fill(iter->first->pt()); 
      v->getMEhisto_Eta()->Fill(iter->first->eta());
      
      DR  = 9999.;
      for ( reco::JetTagCollection::const_iterator iterO = offlineJetTagHandlerPF->begin(); iterO != offlineJetTagHandlerPF->end(); iterO++ )
      { 
        float CSV_offline = iterO->second;
        if (CSV_offline<0) CSV_offline = -0.05;
	DR = reco::deltaR(iterO->first->eta(),iterO->first->phi(),iter->first->eta(),iter->first->phi());
	if (DR<0.3) 
	{
	   v->getMEhisto_CSV_RECOvsHLT()->Fill(CSV_offline,CSV_online); continue;
	   }
	}
	
      iEvent.getByLabel(hltPFPVLabel_, VertexHandler);
      if (VertexHandler.isValid())
      { 
        v->getMEhisto_PVz()->Fill(VertexHandler->begin()->z()); 
        if (offlineVertexHandler.isValid()) v->getMEhisto_PVz_HLTMinusRECO()->Fill(VertexHandler->begin()->z()-offlineVertexHandler->begin()->z());
        }
      }
      
      
     // code dupplication.. needs cleanup..
     //
     if (csvCaloTags.isValid() && v->getTriggerType() == "Calo") 
     { 
      auto iter = csvCaloTags->begin();
      
      //std::cout <<"CALO "<< iter->second <<" " << iter->first->pt() << std::endl;
      
      float CSV_online = iter->second;
      if (CSV_online<0) CSV_online = -0.05;
    
      v->getMEhisto_CSV()->Fill(CSV_online);  
      v->getMEhisto_Pt()->Fill(iter->first->pt()); 
      v->getMEhisto_Eta()->Fill(iter->first->eta());
      
      
      DR  = 9999.;
      for ( reco::JetTagCollection::const_iterator iterO = offlineJetTagHandlerCalo->begin(); iterO != offlineJetTagHandlerCalo->end(); iterO++ )
      {
        float CSV_offline = iterO->second;
        if (CSV_offline<0) CSV_offline = -0.05;
	
        DR = reco::deltaR(iterO->first->eta(),iterO->first->phi(),iter->first->eta(),iter->first->phi());
	if (DR<0.3) 
	{
	   v->getMEhisto_CSV_RECOvsHLT()->Fill(CSV_offline,CSV_online); continue;
	  }  
       }     
      
      iEvent.getByLabel(hltFastPVLabel_, VertexHandler);
      if (VertexHandler.isValid()) 
      {
        v->getMEhisto_PVz()->Fill(VertexHandler->begin()->z()); 
	if (offlineVertexHandler.isValid()) v->getMEhisto_fastPVz_HLTMinusRECO()->Fill(VertexHandler->begin()->z()-offlineVertexHandler->begin()->z());
       }
      
      iEvent.getByLabel(hltCaloPVLabel_, VertexHandler);
      if (VertexHandler.isValid())
      {
        v->getMEhisto_fastPVz()->Fill(VertexHandler->begin()->z()); 
	if (offlineVertexHandler.isValid()) v->getMEhisto_PVz_HLTMinusRECO()->Fill(VertexHandler->begin()->z()-offlineVertexHandler->begin()->z());
       }
       
      }
      
      
    }
   }
  
}


//------------------------------------------------------------------------//
void 
BTVHLTOfflineSource::bookHistograms(DQMStore::IBooker & iBooker, edm::Run const & run, edm::EventSetup const & c)
{
  //-------------------------------------------------
  
  iBooker.setCurrentFolder(dirname_);
    
  //--------------  htlConfig_ ----------------------
    
  bool changed(true);
  if (!hltConfig_.init(run, c, processname_, changed)) {
    LogDebug("BTVHLTOfflineSource") << "HLTConfigProvider failed to initialize.";
  }
   
  //----------- Define hltPathsAll_ -----------------
  
  const unsigned int numberOfPaths(hltConfig_.size());
  for(unsigned int i=0; i!=numberOfPaths; ++i){
    //bool numFound = false;
    pathname_      = hltConfig_.triggerName(i);
    filtername_    = "dummy";
    unsigned int usedPrescale = 1;
    unsigned int objectType = 0;
    std::string triggerType = "";
    bool trigSelected = false;
    
    for (std::vector<std::pair<std::string, std::string> >::iterator custompathnamepair = custompathnamepairs_.begin(); 
          custompathnamepair != custompathnamepairs_.end(); ++custompathnamepair){
       // Checking if the trigger exist in HLT table or not
       if(pathname_.find(custompathnamepair->first)!=std::string::npos) { trigSelected = true; triggerType = custompathnamepair->second;}
      }
    
    if (!trigSelected) continue;
    
    hltPathsAll_.push_back(PathInfo(usedPrescale, pathname_, "dummy", processname_, objectType, triggerType)); 
  
   }//Loop over paths
  

  //------------ book histograns --------------
  
   //
   //std::string dirName = dirname_ + "/MonitorAllTriggers/";
 
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
     CSV->getTH1();
     
     histoname = labelname+"_Pt";
     title = labelname+"_Pt "+trigPath;
     MonitorElement * Pt =  iBooker.book1D(histoname.c_str(),title.c_str(),100,0,400);
     Pt->getTH1();
     
     histoname = labelname+"_Eta";
     title = labelname+"_Eta "+trigPath;
     MonitorElement * Eta =  iBooker.book1D(histoname.c_str(),title.c_str(),60,-3.0,3.0);
     Eta->getTH1();
    
     histoname = "RECOvsHLT_CSV";
     title = "offline CSV vs online CSV "+trigPath;
     MonitorElement * CSV_RECOvsHLT =  iBooker.book2D(histoname.c_str(),title.c_str(),110,-0.1,1,110,-0.1,1);
     CSV_RECOvsHLT->getTH2F();
    
     histoname = labelname+"_PVz";
     title = "online z(PV) "+trigPath;
     MonitorElement * PVz =  iBooker.book1D(histoname.c_str(),title.c_str(),80,-20,20);
     PVz->getTH1();
     
     histoname = labelname+"_fastPVz";
     title = "online z(fastPV) "+trigPath;
     MonitorElement * fastPVz =  iBooker.book1D(histoname.c_str(),title.c_str(),80,-20,20);
     fastPVz->getTH1();
     
     histoname = "HLTMinusRECO_PVz";
     title = "online z(PV) - offline z(PV) "+trigPath;
     MonitorElement * PVz_HLTMinusRECO =  iBooker.book1D(histoname.c_str(),title.c_str(),200,-0.5,0.5);
     PVz_HLTMinusRECO->getTH1();
     
     histoname = "HLTMinusRECO_fastPVz";
     title = "online z(fastPV) - offline z(PV) "+trigPath;
     MonitorElement * fastPVz_HLTMinusRECO =  iBooker.book1D(histoname.c_str(),title.c_str(),100,-2,2);
     fastPVz_HLTMinusRECO->getTH1();
    
     v->setHistos(CSV,Pt,Eta,CSV_RECOvsHLT,PVz,fastPVz,PVz_HLTMinusRECO,fastPVz_HLTMinusRECO);  
   }
         
}


/*
//------------------------------------------------------------------------//
bool BTVHLTOfflineSource::validPathHLT(std::string pathname){
  // hltConfig_ has to be defined first before calling this method
  bool output=false;
  for (unsigned int j=0; j!=hltConfig_.size(); ++j) {
    if (hltConfig_.triggerName(j) == pathname )
      output=true;
  }
  return output;
}

//------------------------------------------------------------------------//
bool BTVHLTOfflineSource::isHLTPathAccepted(std::string pathName){
  // triggerResults_, triggerNames_ has to be defined first before calling this method
  bool output=false;
  if(triggerResults_.isValid()) {
    unsigned index = triggerNames_.triggerIndex(pathName);
    if(index < triggerNames_.size() && triggerResults_->accept(index)) output = true;
  }
  return output;
}

//------------------------------------------------------------------------//
bool BTVHLTOfflineSource::isTriggerObjectFound(std::string objectName){
  // processname_, triggerObj_ has to be defined before calling this method
  bool output=false;
  edm::InputTag testTag(objectName,"",processname_);
  const int index = triggerObj_->filterIndex(testTag);
  if ( index >= triggerObj_->sizeFilters() ) {    
    edm::LogInfo("BTVHLTOfflineSource") << "no index "<< index << " of that name ";
  } else {       
    const trigger::Keys & k = triggerObj_->filterKeys(index);
    if (k.size()) output=true;
  }
  return output;
}
//------------------------------------------------------------------------//
*/
