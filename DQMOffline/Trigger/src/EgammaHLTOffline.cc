#include "DQMOffline/Trigger/interface/EgammaHLTOffline.h"

#include "DQMOffline/Trigger/interface/EleHLTPathMon.h"
#include "DQMOffline/Trigger/interface/EleHLTFilterMon.h"
#include "DQMOffline/Trigger/interface/EgHLTOffData.h"
#include "DQMOffline/Trigger/interface/DebugFuncs.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "FWCore/Framework/interface/TriggerNames.h"
#include "FWCore/Framework/interface/Run.h"

#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

#include "PhysicsTools/Utilities/interface/deltaR.h"

EgammaHLTOffline::EgammaHLTOffline(const edm::ParameterSet& iConfig)
{
  dbe_ = edm::Service<DQMStore>().operator->();
  if (!dbe_) {
    edm::LogInfo("EgammaHLTOffline") << "unable to get DQMStore service?";
  }
  if(iConfig.getUntrackedParameter<bool>("DQMStore", false)) {
    dbe_->setVerbose(0);
  }
  
  triggerSummaryLabel_ = iConfig.getParameter<edm::InputTag>("triggerSummaryLabel");

  
  dirName_=iConfig.getParameter<std::string>("DQMDirName");

  if(dbe_ != 0 ) dbe_->setCurrentFolder("HLTOffline/EgammaHLT");
 
  egHelper_.setup(iConfig);
}


EgammaHLTOffline::~EgammaHLTOffline()
{ 
  // LogDebug("EgammaHLTOffline") << "destructor called";
  for(size_t i=0;i<elePathMonHists_.size();i++){
    delete elePathMonHists_[i];
  }
}

void EgammaHLTOffline::beginJob(const edm::EventSetup& iSetup)
{
  addTrigPath("hltL1NonIsoHLTNonIsoSingleElectronEt15");
  addTrigPath("hltL1NonIsoHLTNonIsoSingleElectronLWEt15");
  
  namesFiltersUsed_.clear();
 filterNamesUsed(namesFiltersUsed_);


}

void EgammaHLTOffline::endJob() 
{
  //  LogDebug("EgammaHLTOffline") << "ending job";
}

void EgammaHLTOffline::beginRun(const edm::Run& run, const edm::EventSetup& c)
{
  //LogDebug("EgammaHLTOffline") << "beginRun, run " << run.id();
}


void EgammaHLTOffline::endRun(const edm::Run& run, const edm::EventSetup& c)
{
  //LogDebug("EgammaHLTOffline") << "endRun, run " << run.id();
}


void EgammaHLTOffline::analyze(const edm::Event& iEvent,const edm::EventSetup& iSetup)
{
  double weight=1.;

  //debugging info, commented out for prod
  // int nrProducts = debug::listAllProducts<trigger::TriggerEvent>(iEvent,"EgammaHLTOffline");
//   edm::LogInfo("EgammaHLTOffline")<<" nr of HLT objs "<<nrProducts;

//   int nrRecHits = debug::listAllProducts<EcalRecHitCollection>(iEvent,"EgammaHLTOffline");
//   edm::LogInfo("EgammaHLTOffline")<<" nr of ecal rec hit collections "<<nrRecHits;


//   int nrEleColl = debug::listAllProducts<reco::PixelMatchGsfElectronCollection>(iEvent,"EgammaHLTOffline");
//   edm::LogInfo("EgammaHLTOffline")<<" nr of Ele coll "<<nrEleColl;

//    int nrClusColl = debug::listAllProducts<reco::BasicClusterShapeAssociationCollection>(iEvent,"EgammaHLTOffline");
//    edm::LogInfo("EgammaHLTOffline")<<" nr of clus "<<nrClusColl;

  edm::Handle<trigger::TriggerEvent> triggerObj;
  iEvent.getByLabel(triggerSummaryLabel_,triggerObj); 
  if(!triggerObj.isValid()) { 
    edm::LogInfo("EgammaHLTOffline") << "Summary HLT objects not found, skipping event"; 
    return;
  }

  edm::Handle<reco::PixelMatchGsfElectronCollection> gsfElectrons;
  iEvent.getByLabel("pixelMatchGsfElectrons",gsfElectrons); 
  if(!gsfElectrons.isValid()) { 
    edm::LogInfo("EgammaHLTOffline") << "gsfElectrons not found,skipping event"; 
    return;
  }  

  egHelper_.getHandles(iEvent,iSetup);

  std::vector<EgHLTOffEle> egHLTOffEles;
  egHelper_.fillEgHLTOffEleVec(gsfElectrons,egHLTOffEles);

  std::vector<std::vector<int> > filtersElePasses;
  obtainFiltersElePasses(egHLTOffEles,namesFiltersUsed_,triggerObj,filtersElePasses);
 
  EgHLTOffData evtData;
  evtData.trigEvt = triggerObj;
  evtData.eles = &egHLTOffEles;
  evtData.filtersElePasses = &filtersElePasses;

//   edm::LogInfo ("EgammaHLTOffline") << "starting event "<<iEvent.id().run()<<" "<<iEvent.id().event();

  
//   edm::LogInfo ("EgammaHLTOffline") << "nr filters in event "<<triggerObj->sizeFilters();
//   for(size_t i=0;i<triggerObj->sizeFilters();i++){
//     edm::LogInfo("EgammaHLTOffline")<<" in event filter "<<triggerObj->filterTag(i);
//   }

  for(size_t pathNr=0;pathNr<elePathMonHists_.size();pathNr++){
    elePathMonHists_[pathNr]->fill(evtData,weight);
  }
}


void EgammaHLTOffline::addTrigPath(std::string name)
{
  EleHLTPathMon* pathMon = new EleHLTPathMon(name);
  pathMon->setStdFilters();
  elePathMonHists_.push_back(pathMon);
  std::sort(elePathMonHists_.begin(),elePathMonHists_.end(),EleHLTFilterMon::ptrLess<EleHLTPathMon>()); //takes a minor efficiency hit at initalisation to ensure that the vector is always sorted
}


void EgammaHLTOffline::obtainFiltersElePasses(const std::vector<EgHLTOffEle>& eles,const std::vector<std::string>& filters,edm::Handle<trigger::TriggerEvent> trigEvt,std::vector<std::vector<int> >& filtersElePasses)
{
  //the swap trick to quickly allocate the vector elements
  std::vector<std::vector<int> > dummyVec(eles.size());
  filtersElePasses.swap(dummyVec);

  const double maxDeltaR=0.3;
  for(size_t filterNrInVec=0;filterNrInVec<filters.size();filterNrInVec++){
    size_t filterNrInEvt = trigEvt->filterIndex(edm::InputTag(filters[filterNrInVec],"","HLT").encode());
    
    if(filterNrInEvt<trigEvt->sizeFilters()){ //filter found in event, something passes it
      const trigger::Keys& trigKeys = trigEvt->filterKeys(filterNrInEvt);  //trigger::Keys is actually a vector<uint16_t> holding the position of trigger objects in the trigger collection passing the filter
      const trigger::TriggerObjectCollection & trigObjColl(trigEvt->getObjects());
      for(size_t eleNr=0;eleNr<eles.size();eleNr++){
	for(trigger::Keys::const_iterator keyIt=trigKeys.begin();keyIt!=trigKeys.end();++keyIt){
	  float trigObjEta = trigObjColl[*keyIt].eta();
	  float trigObjPhi = trigObjColl[*keyIt].phi();
	  if (reco::deltaR(eles[eleNr].eta(),eles[eleNr].phi(),trigObjEta,trigObjPhi) < maxDeltaR){
	    filtersElePasses[eleNr].push_back(filterNrInEvt);
	  }//end dR<0.3 trig obj match test
	}//end loop over all objects passing filter
      }//end loop over electrons
    }//end check if filter is present
  }//end loop over all filters
  
  //now we need to sort the list of trigger numbers that each electron passes
  for(size_t i=0;i<filtersElePasses.size();i++) std::sort(filtersElePasses[i].begin(),filtersElePasses[i].end());
}

void EgammaHLTOffline::filterNamesUsed(std::vector<std::string>& filterNames)
{
  filterNames.clear();
  for(size_t pathNr=0;pathNr<elePathMonHists_.size();pathNr++){
    std::vector<std::string> pathFilters = elePathMonHists_[pathNr]->getFilterNames();
    for(size_t i=0;i<pathFilters.size();i++) filterNames.push_back(pathFilters[i]);
  }
}
