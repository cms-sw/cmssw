#include "DQMOffline/Trigger/interface/EgammaHLTOffline.h"

#include "DQMOffline/Trigger/interface/EleHLTPathMon.h"
#include "DQMOffline/Trigger/interface/EleHLTFilterMon.h"
#include "DQMOffline/Trigger/interface/EgHLTOffData.h"
#include "DQMOffline/Trigger/interface/DebugFuncs.h"
#include "DQMOffline/Trigger/interface/MonElemContainer.h"
#include "DQMOffline/Trigger/interface/EgHLTDQMCut.h"
#include "DQMOffline/Trigger/interface/MonElemFuncs.h"

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
  eleHLTPathNames_ = iConfig.getParameter<std::vector<std::string> >("eleHLTPathNames");
  eleHLTFilterNames_ = iConfig.getParameter<std::vector<std::string> >("eleHLTFilterNames");


  namesFiltersUsed_.clear();
  for(size_t pathNr=0;pathNr<eleHLTPathNames_.size();pathNr++){
    for(size_t filterNr=0;filterNr<eleHLTFilterNames_.size();filterNr++){
      namesFiltersUsed_.push_back(eleHLTPathNames_[pathNr]+eleHLTFilterNames_[filterNr]);
    }
  }
  namesFiltersUsed_.push_back("hltL1NonIsoHLTNonIsoSinglePhotonEt15TrackIsolFilter");
  TrigCodes::setCodes(namesFiltersUsed_);

  dirName_=iConfig.getParameter<std::string>("DQMDirName");//"HLT/EgammaHLTOffline_" + iConfig.getParameter<std::string>("@module_label");

  if(dbe_) dbe_->setCurrentFolder(dirName_);
 
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
  for(size_t i=0;i<eleHLTPathNames_.size();i++) addTrigPath(eleHLTPathNames_[i]);
  
  std::string tightTrig("hltL1NonIsoHLTNonIsoSingleElectronEt15TrackIsolFilter");
  // std::string looseTrig("hltL1NonIsoHLTNonIsoSingleElectronLWEt15TrackIsolFilter");
  std::string looseTrig("hltL1NonIsoHLTNonIsoSinglePhotonEt15TrackIsolFilter");
  
  //single electrons seeing if they pass the tighter trigger
  int stdCutCode = CutCodes::getCode("detEta:crack:sigmaEtaEta:hadem:dPhiIn:dEtaIn"); //will have it non hardcoded at a latter date
  eleMonElems_.push_back(new MonElemContainer<EgHLTOffEle>("passLooseTrig_passTightTrig","",
							   //&(*(new EgMultiCut<EgHLTOffEle>) << 
							   new EgEleTrigCut<EgHLTOffEle>(TrigCodes::getCode(tightTrig+":"+looseTrig),EgEleTrigCut<EgHLTOffEle>::AND)));//  <<
							     //new EgHLTDQMVarCut(stdCutCode,&EgHLTOffEle::cutCode))));

  eleMonElems_.push_back(new MonElemContainer<EgHLTOffEle>("passLooseTrig_failTightTrig","",
							   //&(*(new EgMultiCut<EgHLTOffEle>) << 
							   new EgEleTrigCut<EgHLTOffEle>(TrigCodes::getCode(looseTrig),EgEleTrigCut<EgHLTOffEle>::AND,TrigCodes::getCode(tightTrig))));//  << 
							     //new EgHLTDQMVarCut<EgHLTOffEle>(stdCutCode,&EgHLTOffEle::cutCode))));
  for(size_t i=0;i<eleMonElems_.size();i++){
    MonElemFuncs::initStdEleHists(eleMonElems_[i]->monElems(),tightTrig+"_"+eleMonElems_[i]->name());
  }
  
  //tag and probe trigger efficiencies
  //this is to do measure the trigger efficiency with respect to a fully selected offline electron
  //using a tag and probe technique (note: this will be different to the trigger efficiency normally calculated) 
  for(size_t pathNr=0;pathNr<eleHLTPathNames_.size();pathNr++){
    for(size_t filterNr=0;filterNr<eleHLTFilterNames_.size();filterNr++){ 

      std::string trigName(eleHLTPathNames_[pathNr]+eleHLTFilterNames_[filterNr]);
      int stdCutCode = CutCodes::getCode("detEta:crack:sigmaEtaEta:hadem:dPhiIn:dEtaIn"); //will have it non hardcoded at a latter date
      MonElemContainer<EgHLTOffEle>* monElemCont = new MonElemContainer<EgHLTOffEle>("trigTagProbe","Trigger Tag and Probe",new EgTrigTagProbeCut(TrigCodes::getCode(trigName),stdCutCode,&EgHLTOffEle::cutCode));
      MonElemFuncs::initStdEleCutHists(monElemCont->cutMonElems(),trigName+"_"+monElemCont->name()+"_all",new EgGreaterCut<EgHLTOffEle,float>(15.,&EgHLTOffEle::etSC));
      MonElemFuncs::initStdEleCutHists(monElemCont->cutMonElems(),trigName+"_"+monElemCont->name()+"_pass",&(*(new EgMultiCut<EgHLTOffEle>) << new EgGreaterCut<EgHLTOffEle,float>(15.,&EgHLTOffEle::etSC) << new EgEleTrigCut<EgHLTOffEle>(TrigCodes::getCode(trigName),EgEleTrigCut<EgHLTOffEle>::AND)));
      eleMonElems_.push_back(monElemCont);
    } //end filter names
  }//end path names
  //namesFiltersUsed_.clear();
  //filterNamesUsed(namesFiltersUsed_);

 //defines which bits are assoicated to which filter
 
  // TrigCodes::printCodes();

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
  const double weight=1.;

  //debugging info, commented out for prod
  //int nrProducts = debug::listAllProducts<reco::CaloJetCollection>(iEvent,"EgammaHLTOffline");
  //  edm::LogInfo("EgammaHLTOffline")<<" nr of jet obs "<<nrProducts;

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
  TrigCodes::TrigBitSet evtTrigBits = setFiltersElePasses(egHLTOffEles,namesFiltersUsed_,triggerObj);
 
  EgHLTOffData evtData;
  evtData.trigEvt = triggerObj;
  evtData.eles = &egHLTOffEles;
  evtData.filtersElePasses = &filtersElePasses;
  evtData.evtTrigBits = evtTrigBits;
  evtData.jets = egHelper_.jets();
  
//   edm::LogInfo ("EgammaHLTOffline") << "starting event "<<iEvent.id().run()<<" "<<iEvent.id().event();

  //edm::LogInfo ("EgammaHLTOffline") << "nr filters in event "<<triggerObj->sizeFilters();
     //for(size_t i=0;i<triggerObj->sizeFilters();i++){
     //edm::LogInfo("EgammaHLTOffline")<<" in event filter "<<triggerObj->filterTag(i);
     //}

  for(size_t pathNr=0;pathNr<elePathMonHists_.size();pathNr++){
    elePathMonHists_[pathNr]->fill(evtData,weight);
  }

  for(size_t monElemNr=0;monElemNr<eleMonElems_.size();monElemNr++){
    const std::vector<EgHLTOffEle>& eles = *evtData.eles;
    for(size_t eleNr=0;eleNr<eles.size();eleNr++){
      eleMonElems_[monElemNr]->fill(eles[eleNr],evtData,weight);
    }
  }
}


void EgammaHLTOffline::addTrigPath(const std::string& name)
{
  EleHLTPathMon* pathMon = new EleHLTPathMon(name);
  pathMon->addFilters(eleHLTFilterNames_);
  elePathMonHists_.push_back(pathMon);
  std::sort(elePathMonHists_.begin(),elePathMonHists_.end(),EleHLTFilterMon::ptrLess<EleHLTPathMon>()); //takes a minor efficiency hit at initalisation to ensure that the vector is always sorted
}

//I have the horrible feeling that I'm converting into an intermediatry format and then coverting back again
//Okay how this works
//1) create a TrigBitSet for each electron set to 0 initally
//2) loop over each filter, for each electron that passes the filter, set the appropriate bit in the TrigBitSet
//3) after that, loop over each electron setting the its TrigBitSet which has been calculated
//4) a crowbar hack now has it also create a bitset for all the triggers which fired in the event (which are in the filters vector). Note these dont have to be electron triggers
TrigCodes::TrigBitSet EgammaHLTOffline::setFiltersElePasses(std::vector<EgHLTOffEle>& eles,const std::vector<std::string>& filters,edm::Handle<trigger::TriggerEvent> trigEvt)
{
  TrigCodes::TrigBitSet evtTrigs;
  std::vector<TrigCodes::TrigBitSet> eleTrigBits(eles.size());
  const double maxDeltaR=0.3;
  for(size_t filterNrInVec=0;filterNrInVec<filters.size();filterNrInVec++){
    size_t filterNrInEvt = trigEvt->filterIndex(edm::InputTag(filters[filterNrInVec],"","HLT").encode());
    const TrigCodes::TrigBitSet filterCode = TrigCodes::getCode(filters[filterNrInVec].c_str());

    if(filterNrInEvt<trigEvt->sizeFilters()){ //filter found in event, something passes it
      evtTrigs |=filterCode; //if something passes it add to the event trigger bits
      const trigger::Keys& trigKeys = trigEvt->filterKeys(filterNrInEvt);  //trigger::Keys is actually a vector<uint16_t> holding the position of trigger objects in the trigger collection passing the filter
      const trigger::TriggerObjectCollection & trigObjColl(trigEvt->getObjects());
      for(size_t eleNr=0;eleNr<eles.size();eleNr++){
	for(trigger::Keys::const_iterator keyIt=trigKeys.begin();keyIt!=trigKeys.end();++keyIt){
	  float trigObjEta = trigObjColl[*keyIt].eta();
	  float trigObjPhi = trigObjColl[*keyIt].phi();
	  if (reco::deltaR(eles[eleNr].eta(),eles[eleNr].phi(),trigObjEta,trigObjPhi) < maxDeltaR){
	    eleTrigBits[eleNr] |= filterCode;
	  }//end dR<0.3 trig obj match test
	}//end loop over all objects passing filter
      }//end loop over electrons
    }//end check if filter is present
  }//end loop over all filters

  for(size_t eleNr=0;eleNr<eles.size();eleNr++) eles[eleNr].setTrigBits(eleTrigBits[eleNr]);

  return evtTrigs;

}



//TriggerEvent will for each filter name tell you the index it has in the event
//I can then use this to get a list of candidates which pass the trigger and I do a deltaR match
//to figure out if it corrsponds to the vincity of my electron (there *must* be a better way) 
//eles: list of electrons in event
//filters: list of filter names I want to check if the electron passes eg hltL1NonIsoHLTNonIsoSingleElectronEt15TrackIsolFilter
//trigEvt : the handle to the TriggerEvent
//filtersElePasses: a series of vectors, each vector corresponding to an electron. Each vector has a list of filter names the electron passes
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
