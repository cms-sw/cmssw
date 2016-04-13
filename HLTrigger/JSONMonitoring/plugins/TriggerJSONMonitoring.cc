/** \class TriggerJSONMonitoring
 *  
 * See header file for documentation
 *
 * 
 *  \author Aram Avetisyan
 *  \author Daniel Salerno 
 * 
 */

#include "HLTrigger/JSONMonitoring/interface/TriggerJSONMonitoring.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "EventFilter/Utilities/interface/FastMonitoringService.h"

#include <fstream>
using namespace jsoncollector;

TriggerJSONMonitoring::TriggerJSONMonitoring(const edm::ParameterSet& ps) :
  triggerResults_(ps.getParameter<edm::InputTag>("triggerResults")),
  triggerResultsToken_(consumes<edm::TriggerResults>(triggerResults_)),
  level1Results_(ps.getParameter<edm::InputTag>("L1Results")),   
  m_l1t_results(consumes<GlobalAlgBlkBxCollection>(level1Results_))             
{

                                                     
}

TriggerJSONMonitoring::~TriggerJSONMonitoring()
{
}

void
TriggerJSONMonitoring::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("triggerResults",edm::InputTag("TriggerResults","","HLT"));
  desc.add<edm::InputTag>("L1Results",edm::InputTag("hltGtStage2Digis"));                
  descriptions.add("triggerJSONMonitoring", desc);
}

void
TriggerJSONMonitoring::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  using namespace std;
  using namespace edm;

  processed_++;

  int ex = iEvent.experimentType();
  if      (ex == 1) L1Global_[0]++; 
  else if (ex == 2) L1Global_[1]++;
  else if (ex == 3) L1Global_[2]++;
  else{
    LogDebug("TriggerJSONMonitoring") << "Not Physics, Calibration or Random. experimentType = " << ex << std::endl;
  }   

  //Temporarily removing L1 monitoring while we adapt for Stage 2
  //Get hold of L1TResults 
  // edm::Handle<L1GlobalTriggerReadoutRecord> l1tResults;
  // iEvent.getByToken(m_l1t_results, l1tResults);

  // L1GlobalTriggerReadoutRecord L1TResults = * l1tResults.product();

  // const std::vector<bool> & algoword = L1TResults.decisionWord();  
  // if (algoword.size() == L1AlgoAccept_.size()){
  //   for (unsigned int i = 0; i < algoword.size(); i++){
  //     if (algoword[i]){
  // 	L1AlgoAccept_[i]++;
  // 	if (ex == 1) L1AlgoAcceptPhysics_[i]++;
  // 	if (ex == 2) L1AlgoAcceptCalibration_[i]++;
  // 	if (ex == 3) L1AlgoAcceptRandom_[i]++;
  //     }
  //   }
  // }
  // else {
  //   LogWarning("TriggerJSONMonitoring")<<"L1 Algo Trigger Mask size does not match number of L1 Algo Triggers!";
  // }

  // const std::vector<bool> & techword = L1TResults.technicalTriggerWord();
  // if (techword.size() == L1TechAccept_.size()){
  //   for (unsigned int i = 0; i < techword.size(); i++){
  //     if (techword[i]){
  // 	L1TechAccept_[i]++;
  // 	if (ex == 1) L1TechAcceptPhysics_[i]++;
  // 	if (ex == 2) L1TechAcceptCalibration_[i]++;
  // 	if (ex == 3) L1TechAcceptRandom_[i]++;
  //     }
  //   }
  // }
  // else{
  //   LogWarning("TriggerJSONMonitoring")<<"L1 Tech Trigger Mask size does not match number of L1 Tech Triggers!";
  // }
  
  //Get hold of TriggerResults  
  Handle<TriggerResults> HLTR;
  iEvent.getByToken(triggerResultsToken_, HLTR);
  if (!HLTR.isValid()) {
    LogDebug("TriggerJSONMonitoring") << "HLT TriggerResults with label ["+triggerResults_.encode()+"] not found!" << std::endl;
    return;
  }

  //Decision for each HLT path   
  const unsigned int n(hltNames_.size());
  for (unsigned int i=0; i<n; i++) {
    if (HLTR->wasrun(i))                     hltWasRun_[i]++;
    if (HLTR->accept(i))                     hltAccept_[i]++;
    if (HLTR->wasrun(i) && !HLTR->accept(i)) hltReject_[i]++;
    if (HLTR->error(i))                      hltErrors_[i]++;
    //Count L1 seeds and Prescales   
    const int index(static_cast<int>(HLTR->index(i)));
    if (HLTR->accept(i)) {
      if (index >= posL1s_[i]) hltL1s_[i]++;
      if (index >= posPre_[i]) hltPre_[i]++;
    } else {
      if (index >  posL1s_[i]) hltL1s_[i]++;
      if (index >  posPre_[i]) hltPre_[i]++;
    }
  }

  //Decision for each HLT dataset     
  std::vector<bool> acceptedByDS(hltIndex_.size(), false);
  for (unsigned int ds=0; ds < hltIndex_.size(); ds++) {  // ds = index of dataset       
    for (unsigned int p=0; p<hltIndex_[ds].size(); p++) {   // p = index of path with dataset ds       
      if (acceptedByDS[ds]>0 || HLTR->accept(hltIndex_[ds][p]) ) {
	acceptedByDS[ds] = true;
      }
    }
    if (acceptedByDS[ds]) hltDatasets_[ds]++;
  }

  //Prescale index
  edm::Handle<GlobalAlgBlkBxCollection> l1tResults;
  if (iEvent.getByToken(m_l1t_results, l1tResults) and not (l1tResults->begin(0) == l1tResults->end(0)))
    prescaleIndex_ = static_cast<unsigned int>(l1tResults->begin(0)->getPreScColumn());
  
  //Check that the prescale index hasn't changed inside a lumi section
  unsigned int newLumi = (unsigned int) iEvent.eventAuxiliary().luminosityBlock();
  if (oldLumi == newLumi and prescaleIndex_ != oldPrescaleIndex){
    LogWarning("TriggerJSONMonitoring")<<"Prescale index has changed from "<<oldPrescaleIndex<<" to "<<prescaleIndex_<<" inside lumi section "<<newLumi;
  }
  oldLumi = newLumi;
  oldPrescaleIndex = prescaleIndex_;

}//End analyze function     

void
TriggerJSONMonitoring::resetRun(bool changed){

  //Update trigger and dataset names, clear L1 names and counters   
  if (changed){
    hltNames_        = hltConfig_.triggerNames();
    datasetNames_    = hltConfig_.datasetNames();
    datasetContents_ = hltConfig_.datasetContents();

    L1AlgoNames_.resize(m_l1tAlgoMask->gtTriggerMask().size());         
    for (unsigned int i = 0; i < L1AlgoNames_.size(); i++) {
      L1AlgoNames_.at(i) = "";
    }
    //Get L1 algorithm trigger names -      
    for (CItAlgo itAlgo = algorithmMap.begin(); itAlgo != algorithmMap.end(); itAlgo++) {
      int bitNumber = (itAlgo->second).algoBitNumber();
      L1AlgoNames_.at(bitNumber) = itAlgo->first;
    }

    L1TechNames_.resize(m_l1tTechMask->gtTriggerMask().size());    
    for (unsigned int i = 0; i < L1TechNames_.size(); i++) {
      L1TechNames_.at(i) = "";
    }
    //Get L1 technical trigger names -           
    for (CItAlgo itAlgo = technicalMap.begin(); itAlgo != technicalMap.end(); itAlgo++) {
      int bitNumber = (itAlgo->second).algoBitNumber();
      L1TechNames_.at(bitNumber) = itAlgo->first;
    }

    L1GlobalType_.clear();   
    L1Global_.clear();     
    
    //Set the experimentType -          
    L1GlobalType_.push_back( "Physics" );
    L1GlobalType_.push_back( "Calibration" );
    L1GlobalType_.push_back( "Random" );
  }

  const unsigned int n  = hltNames_.size();
  const unsigned int d  = datasetNames_.size();
  const unsigned int la = L1AlgoNames_.size();       
  const unsigned int lt = L1TechNames_.size();     
  const unsigned int lg = L1GlobalType_.size();      

  if (changed) {
    //Resize per-path counters   
    hltWasRun_.resize(n);
    hltL1s_.resize(n);
    hltPre_.resize(n);
    hltAccept_.resize(n);
    hltReject_.resize(n);
    hltErrors_.resize(n);

    L1AlgoAccept_.resize(la);         
    L1AlgoAcceptPhysics_.resize(la);         
    L1AlgoAcceptCalibration_.resize(la);         
    L1AlgoAcceptRandom_.resize(la);         

    L1TechAccept_.resize(lt);          
    L1TechAcceptPhysics_.resize(lt);          
    L1TechAcceptCalibration_.resize(lt);          
    L1TechAcceptRandom_.resize(lt);          

    L1Global_.resize(lg);                 
    //Resize per-dataset counter    
    hltDatasets_.resize(d);
    //Resize htlIndex     
    hltIndex_.resize(d);
    //Set-up hltIndex          
    for (unsigned int ds = 0; ds < d; ds++) {
      unsigned int size = datasetContents_[ds].size();
      hltIndex_[ds].reserve(size);
      for (unsigned int p = 0; p < size; p++) {
	unsigned int i = hltConfig_.triggerIndex(datasetContents_[ds][p]);
	if (i<n) {
	  hltIndex_[ds].push_back(i);
	}
      }
    }
    //Find the positions of seeding and prescaler modules     
    posL1s_.resize(n);
    posPre_.resize(n);
    for (unsigned int i = 0; i < n; ++i) {
      posL1s_[i] = -1;
      posPre_[i] = -1;
      const std::vector<std::string> & moduleLabels(hltConfig_.moduleLabels(i));
      for (unsigned int j = 0; j < moduleLabels.size(); ++j) {
	const std::string & label = hltConfig_.moduleType(moduleLabels[j]);
	if (label == "HLTLevel1GTSeed")
	  posL1s_[i] = j;
	else if (label == "HLTPrescaler")
	  posPre_[i] = j;
      }
    }
  }
  resetLumi();
}//End resetRun function                  

void
TriggerJSONMonitoring::resetLumi(){
  //Reset total number of events     
  processed_ = 0;

  //Reset per-path counters 
  for (unsigned int i = 0; i < hltWasRun_.size(); i++) {
    hltWasRun_[i] = 0;
    hltL1s_[i]    = 0;
    hltPre_[i]    = 0;
    hltAccept_[i] = 0;
    hltReject_[i] = 0;
    hltErrors_[i] = 0;
  }
  //Reset per-dataset counter         
  for (unsigned int i = 0; i < hltDatasets_.size(); i++) {
    hltDatasets_[i] = 0;
  }
  //Reset L1 per-algo counters -     
  for (unsigned int i = 0; i < L1AlgoAccept_.size(); i++) {
    L1AlgoAccept_[i]            = 0;
    L1AlgoAcceptPhysics_[i]     = 0;
    L1AlgoAcceptCalibration_[i] = 0;
    L1AlgoAcceptRandom_[i]      = 0;
  }
  //Reset L1 per-tech counters -    
  for (unsigned int i = 0; i < L1TechAccept_.size(); i++) {
    L1TechAccept_[i]            = 0;
    L1TechAcceptPhysics_[i]     = 0;
    L1TechAcceptCalibration_[i] = 0;
    L1TechAcceptRandom_[i]      = 0;
  }
  //Reset L1 global counters -      
  for (unsigned int i = 0; i < L1GlobalType_.size(); i++) {
    L1Global_[i] = 0;
  }

  //Luminosity and prescale index
  prescaleIndex_ = 0;

}//End resetLumi function  

void
TriggerJSONMonitoring::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup)
{
  //Get the run directory from the EvFDaqDirector                
  if (edm::Service<evf::EvFDaqDirector>().isAvailable()) baseRunDir_ = edm::Service<evf::EvFDaqDirector>()->baseRunDir();
  else                                                   baseRunDir_ = ".";

  std::string monPath = baseRunDir_ + "/";

  //Get/update the L1 trigger menu from the EventSetup
  edm::ESHandle<L1GtTriggerMenu> l1GtMenu;           
  iSetup.get<L1GtTriggerMenuRcd>().get(l1GtMenu);    
  m_l1GtMenu = l1GtMenu.product();                   
  algorithmMap = m_l1GtMenu->gtAlgorithmMap();       
  technicalMap = m_l1GtMenu->gtTechnicalTriggerMap();

  //Get masks (for now, only use them to find the number of triggers)
  edm::ESHandle<L1GtTriggerMask> l1GtAlgoMask;           
  iSetup.get<L1GtTriggerMaskAlgoTrigRcd>().get(l1GtAlgoMask);    
  m_l1tAlgoMask = l1GtAlgoMask.product();                   

  edm::ESHandle<L1GtTriggerMask> l1GtTechMask;           
  iSetup.get<L1GtTriggerMaskTechTrigRcd>().get(l1GtTechMask);    
  m_l1tTechMask = l1GtTechMask.product();                   

  //Initialize hltConfig_     
  bool changed = true;
  if (hltConfig_.init(iRun, iSetup, triggerResults_.process(), changed)) resetRun(changed);
  else{
    LogDebug("TriggerJSONMonitoring") << "HLTConfigProvider initialization failed!" << std::endl;
    return;
  }

  //Write the once-per-run files if not already written
  //Eventually must rewrite this with proper multithreading (i.e. globalBeginRun)
  bool expected = false;
  if( runCache()->wroteFiles.compare_exchange_strong(expected, true) ){
    runCache()->wroteFiles = true;

    unsigned int nRun = iRun.run();
    
    //Create definition file for HLT Rates                 
    std::stringstream ssHltJsd;
    ssHltJsd << "run" << std::setfill('0') << std::setw(6) << nRun << "_ls0000";
    ssHltJsd << "_streamHLTRates_pid" << std::setfill('0') << std::setw(5) << getpid() << ".jsd";
    stHltJsd_ = ssHltJsd.str();

    writeDefJson(baseRunDir_ + "/" + stHltJsd_);
    
    //Create definition file for L1 Rates -  
    std::stringstream ssL1Jsd;
    ssL1Jsd << "run" << std::setfill('0') << std::setw(6) << nRun << "_ls0000";
    ssL1Jsd << "_streamL1Rates_pid" << std::setfill('0') << std::setw(5) << getpid() << ".jsd";
    stL1Jsd_ = ssL1Jsd.str();

    writeL1DefJson(baseRunDir_ + "/" + stL1Jsd_);
    
    //Write ini files
    //HLT
    Json::Value hltIni;
    Json::StyledWriter writer;

    Json::Value hltNamesVal(Json::arrayValue);
    for (unsigned int ui = 0; ui < hltNames_.size(); ui++){
      hltNamesVal.append(hltNames_.at(ui));
    }

    Json::Value datasetNamesVal(Json::arrayValue);
    for (unsigned int ui = 0; ui < datasetNames_.size(); ui++){
      datasetNamesVal.append(datasetNames_.at(ui));
    }

    hltIni["Path-Names"]    = hltNamesVal;
    hltIni["Dataset-Names"] = datasetNamesVal;
    
    std::string && result = writer.write(hltIni);
  
    std::stringstream ssHltIni;
    ssHltIni << "run" << std::setfill('0') << std::setw(6) << nRun << "_ls0000_streamHLTRates_pid" << std::setfill('0') << std::setw(5) << getpid() << ".ini";
    
    std::ofstream outHltIni( monPath + ssHltIni.str() );
    outHltIni<<result;
    outHltIni.close();
    
    //L1
    Json::Value l1Ini;

    Json::Value l1AlgoNamesVal(Json::arrayValue);
    for (unsigned int ui = 0; ui < L1AlgoNames_.size(); ui++){
      l1AlgoNamesVal.append(L1AlgoNames_.at(ui));
    }

    Json::Value l1TechNamesVal(Json::arrayValue);
    for (unsigned int ui = 0; ui < L1TechNames_.size(); ui++){
      l1TechNamesVal.append(L1TechNames_.at(ui));
    }

    Json::Value eventTypeVal(Json::arrayValue);
    for (unsigned int ui = 0; ui < L1GlobalType_.size(); ui++){
      eventTypeVal.append(L1GlobalType_.at(ui));
    }

    l1Ini["L1-Algo-Names"] = l1AlgoNamesVal;
    l1Ini["L1-Tech-Names"] = l1TechNamesVal;
    l1Ini["Event-Type"]    = eventTypeVal;
    
    result = writer.write(l1Ini);
  
    std::stringstream ssL1Ini;
    ssL1Ini << "run" << std::setfill('0') << std::setw(6) << nRun << "_ls0000_streamL1Rates_pid" << std::setfill('0') << std::setw(5) << getpid() << ".ini";
    
    std::ofstream outL1Ini( monPath + ssL1Ini.str() );
    outL1Ini<<result;
    outL1Ini.close();
  }

  //Initialize variables for verification of prescaleIndex
  oldLumi          = 0;
  oldPrescaleIndex = 0;

}//End beginRun function        

void TriggerJSONMonitoring::beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup& iSetup){ resetLumi(); }

std::shared_ptr<trigJson::lumiVars>
TriggerJSONMonitoring::globalBeginLuminosityBlockSummary(const edm::LuminosityBlock& iLumi, const edm::EventSetup& iSetup, const LuminosityBlockContext* iContext)
{
  std::shared_ptr<trigJson::lumiVars> iSummary(new trigJson::lumiVars);

  unsigned int MAXPATHS = 500;

  iSummary->processed = new HistoJ<unsigned int>(1, 1);

  iSummary->hltWasRun = new HistoJ<unsigned int>(1, MAXPATHS);
  iSummary->hltL1s    = new HistoJ<unsigned int>(1, MAXPATHS);
  iSummary->hltPre    = new HistoJ<unsigned int>(1, MAXPATHS);
  iSummary->hltAccept = new HistoJ<unsigned int>(1, MAXPATHS);
  iSummary->hltReject = new HistoJ<unsigned int>(1, MAXPATHS);
  iSummary->hltErrors = new HistoJ<unsigned int>(1, MAXPATHS);

  iSummary->hltDatasets = new HistoJ<unsigned int>(1, MAXPATHS);

  iSummary->prescaleIndex = 100;

  iSummary->L1AlgoAccept            = new HistoJ<unsigned int>(1, MAXPATHS);                            
  iSummary->L1TechAccept            = new HistoJ<unsigned int>(1, MAXPATHS);  
  iSummary->L1AlgoAcceptPhysics     = new HistoJ<unsigned int>(1, MAXPATHS);                            
  iSummary->L1TechAcceptPhysics     = new HistoJ<unsigned int>(1, MAXPATHS);  
  iSummary->L1AlgoAcceptCalibration = new HistoJ<unsigned int>(1, MAXPATHS);                            
  iSummary->L1TechAcceptCalibration = new HistoJ<unsigned int>(1, MAXPATHS);  
  iSummary->L1AlgoAcceptRandom      = new HistoJ<unsigned int>(1, MAXPATHS);                            
  iSummary->L1TechAcceptRandom      = new HistoJ<unsigned int>(1, MAXPATHS);  
  iSummary->L1Global                = new HistoJ<unsigned int>(1, MAXPATHS);  

  iSummary->baseRunDir           = "";
  iSummary->stHltJsd             = "";
  iSummary->stL1Jsd              = "";
  iSummary->streamL1Destination  = "";
  iSummary->streamHLTDestination = "";

  return iSummary;
}//End globalBeginLuminosityBlockSummary function  

void
TriggerJSONMonitoring::endLuminosityBlockSummary(const edm::LuminosityBlock& iLumi, const edm::EventSetup& iEventSetup, trigJson::lumiVars* iSummary) const{

  //Whichever stream gets there first does the initialiazation 
  if (iSummary->hltWasRun->value().size() == 0){
    iSummary->processed->update(processed_);

    for (unsigned int ui = 0; ui < hltWasRun_.size(); ui++){
      iSummary->hltWasRun->update(hltWasRun_.at(ui));
      iSummary->hltL1s   ->update(hltL1s_   .at(ui));
      iSummary->hltPre   ->update(hltPre_   .at(ui));
      iSummary->hltAccept->update(hltAccept_.at(ui));
      iSummary->hltReject->update(hltReject_.at(ui));
      iSummary->hltErrors->update(hltErrors_.at(ui));
    }
    for (unsigned int ui = 0; ui < hltDatasets_.size(); ui++){
      iSummary->hltDatasets->update(hltDatasets_.at(ui));
    }
    iSummary->prescaleIndex = prescaleIndex_;

    iSummary->stHltJsd   = stHltJsd_;
    iSummary->baseRunDir = baseRunDir_;
    
    for (unsigned int ui = 0; ui < L1AlgoAccept_.size(); ui++){    
      iSummary->L1AlgoAccept           ->update(L1AlgoAccept_.at(ui));
      iSummary->L1AlgoAcceptPhysics    ->update(L1AlgoAcceptPhysics_.at(ui));
      iSummary->L1AlgoAcceptCalibration->update(L1AlgoAcceptCalibration_.at(ui));
      iSummary->L1AlgoAcceptRandom     ->update(L1AlgoAcceptRandom_.at(ui));
    }
    for (unsigned int ui = 0; ui < L1TechAccept_.size(); ui++){   
      iSummary->L1TechAccept           ->update(L1TechAccept_.at(ui));
      iSummary->L1TechAcceptPhysics    ->update(L1TechAcceptPhysics_.at(ui));
      iSummary->L1TechAcceptCalibration->update(L1TechAcceptCalibration_.at(ui));
      iSummary->L1TechAcceptRandom     ->update(L1TechAcceptRandom_.at(ui));
    }
    for (unsigned int ui = 0; ui < L1GlobalType_.size(); ui++){    
      iSummary->L1Global    ->update(L1Global_.at(ui));
    }
    iSummary->stL1Jsd = stL1Jsd_;      

    iSummary->streamHLTDestination = runCache()->streamHLTDestination;
    iSummary->streamL1Destination  = runCache()->streamL1Destination;
  }

  else{
    iSummary->processed->value().at(0) += processed_;

    for (unsigned int ui = 0; ui < hltWasRun_.size(); ui++){
      iSummary->hltWasRun->value().at(ui) += hltWasRun_.at(ui);
      iSummary->hltL1s   ->value().at(ui) += hltL1s_   .at(ui);
      iSummary->hltPre   ->value().at(ui) += hltPre_   .at(ui);
      iSummary->hltAccept->value().at(ui) += hltAccept_.at(ui);
      iSummary->hltReject->value().at(ui) += hltReject_.at(ui);
      iSummary->hltErrors->value().at(ui) += hltErrors_.at(ui);
    }
    for (unsigned int ui = 0; ui < hltDatasets_.size(); ui++){
      iSummary->hltDatasets->value().at(ui) += hltDatasets_.at(ui);
    }
    for (unsigned int ui = 0; ui < L1AlgoAccept_.size(); ui++){                             
      iSummary->L1AlgoAccept->value().at(ui)            += L1AlgoAccept_.at(ui);
      iSummary->L1AlgoAcceptPhysics->value().at(ui)     += L1AlgoAcceptPhysics_.at(ui);
      iSummary->L1AlgoAcceptCalibration->value().at(ui) += L1AlgoAcceptCalibration_.at(ui);
      iSummary->L1AlgoAcceptRandom->value().at(ui)      += L1AlgoAcceptRandom_.at(ui);
    }
    for (unsigned int ui = 0; ui < L1TechAccept_.size(); ui++){                                
      iSummary->L1TechAccept->value().at(ui)            += L1TechAccept_.at(ui);
      iSummary->L1TechAcceptPhysics->value().at(ui)     += L1TechAcceptPhysics_.at(ui);
      iSummary->L1TechAcceptCalibration->value().at(ui) += L1TechAcceptCalibration_.at(ui);
      iSummary->L1TechAcceptRandom->value().at(ui)      += L1TechAcceptRandom_.at(ui);
    }
    for (unsigned int ui = 0; ui < L1Global_.size(); ui++){                               
      iSummary->L1Global->value().at(ui) += L1Global_.at(ui);
    }

  }

}//End endLuminosityBlockSummary function                                             


void
TriggerJSONMonitoring::globalEndLuminosityBlockSummary(const edm::LuminosityBlock& iLumi, const edm::EventSetup& iSetup, const LuminosityBlockContext* iContext, trigJson::lumiVars* iSummary)
{

  unsigned int iLs  = iLumi.luminosityBlock();
  unsigned int iRun = iLumi.run();

  bool writeFiles=true;
  if (edm::Service<evf::MicroStateService>().isAvailable()) {
    evf::FastMonitoringService * fms = (evf::FastMonitoringService *)(edm::Service<evf::MicroStateService>().operator->());
    if (fms) {
      writeFiles = fms->shouldWriteFiles(iLumi.luminosityBlock());
    }
  }

  if (writeFiles) {
    Json::StyledWriter writer;

    char hostname[33];
    gethostname(hostname,32);
    std::string sourceHost(hostname);

    //Get the output directory                                        
    std::string monPath = iSummary->baseRunDir + "/";

    std::stringstream sOutDef;
    sOutDef << monPath << "output_" << getpid() << ".jsd";

    //Write the .jsndata files which contain the actual rates
    //HLT .jsndata file
    Json::Value hltJsnData;
    hltJsnData[DataPoint::SOURCE] = sourceHost;
    hltJsnData[DataPoint::DEFINITION] = iSummary->stHltJsd;

    hltJsnData[DataPoint::DATA].append(iSummary->processed->toJsonValue());
    hltJsnData[DataPoint::DATA].append(iSummary->hltWasRun->toJsonValue());
    hltJsnData[DataPoint::DATA].append(iSummary->hltL1s   ->toJsonValue());
    hltJsnData[DataPoint::DATA].append(iSummary->hltPre   ->toJsonValue());
    hltJsnData[DataPoint::DATA].append(iSummary->hltAccept->toJsonValue());
    hltJsnData[DataPoint::DATA].append(iSummary->hltReject->toJsonValue());
    hltJsnData[DataPoint::DATA].append(iSummary->hltErrors->toJsonValue());

    hltJsnData[DataPoint::DATA].append(iSummary->hltDatasets->toJsonValue());

    hltJsnData[DataPoint::DATA].append(iSummary->prescaleIndex);

    std::string && result = writer.write(hltJsnData);

    std::stringstream ssHltJsnData;
    ssHltJsnData <<  "run" << std::setfill('0') << std::setw(6) << iRun << "_ls" << std::setfill('0') << std::setw(4) << iLs;
    ssHltJsnData << "_streamHLTRates_pid" << std::setfill('0') << std::setw(5) << getpid() << ".jsndata";

    if (iSummary->processed->value().at(0)!=0) {
      std::ofstream outHltJsnData( monPath + ssHltJsnData.str() );
      outHltJsnData<<result;
      outHltJsnData.close();
    }

    //HLT jsn entries
    StringJ hltJsnFilelist;
    IntJ hltJsnFilesize    = 0;
    unsigned int hltJsnFileAdler32 = 1;
    if (iSummary->processed->value().at(0)!=0) {
      hltJsnFilelist.update(ssHltJsnData.str());
      hltJsnFilesize    = result.size();
      hltJsnFileAdler32 = cms::Adler32(result.c_str(),result.size());
    }
    StringJ hltJsnInputFiles;
    hltJsnInputFiles.update("");

    //L1 .jsndata file
    Json::Value l1JsnData;
    l1JsnData[DataPoint::SOURCE] = sourceHost;
    l1JsnData[DataPoint::DEFINITION] = iSummary->stL1Jsd;

    l1JsnData[DataPoint::DATA].append(iSummary->processed->toJsonValue());
    l1JsnData[DataPoint::DATA].append(iSummary->L1AlgoAccept           ->toJsonValue());
    l1JsnData[DataPoint::DATA].append(iSummary->L1TechAccept           ->toJsonValue());        
    l1JsnData[DataPoint::DATA].append(iSummary->L1AlgoAcceptPhysics    ->toJsonValue());
    l1JsnData[DataPoint::DATA].append(iSummary->L1TechAcceptPhysics    ->toJsonValue());        
    l1JsnData[DataPoint::DATA].append(iSummary->L1AlgoAcceptCalibration->toJsonValue());
    l1JsnData[DataPoint::DATA].append(iSummary->L1TechAcceptCalibration->toJsonValue());        
    l1JsnData[DataPoint::DATA].append(iSummary->L1AlgoAcceptRandom     ->toJsonValue());
    l1JsnData[DataPoint::DATA].append(iSummary->L1TechAcceptRandom     ->toJsonValue());        
    l1JsnData[DataPoint::DATA].append(iSummary->L1Global               ->toJsonValue());      
    result = writer.write(l1JsnData);

    std::stringstream ssL1JsnData;
    ssL1JsnData << "run" << std::setfill('0') << std::setw(6) << iRun << "_ls" << std::setfill('0') << std::setw(4) << iLs;
    ssL1JsnData << "_streamL1Rates_pid" << std::setfill('0') << std::setw(5) << getpid() << ".jsndata";

    if (iSummary->processed->value().at(0)!=0) {
      std::ofstream outL1JsnData( monPath + "/" + ssL1JsnData.str() );
      outL1JsnData<<result;
      outL1JsnData.close();
    }

    //L1 jsn entries
    StringJ l1JsnFilelist;
    IntJ l1JsnFilesize    = 0;
    unsigned int l1JsnFileAdler32 = 1;
    if (iSummary->processed->value().at(0)!=0) {
      l1JsnFilelist.update(ssL1JsnData.str());
      l1JsnFilesize    = result.size();
      l1JsnFileAdler32 = cms::Adler32(result.c_str(),result.size());
    }
    StringJ l1JsnInputFiles;
    l1JsnInputFiles.update("");


    //Create special DAQ JSON file for L1 and HLT rates pseudo-streams
    //Only three variables are different between the files: 
    //the file list, the file size and the Adler32 value
    IntJ daqJsnProcessed   = iSummary->processed->value().at(0);
    IntJ daqJsnAccepted    = daqJsnProcessed;
    IntJ daqJsnErrorEvents = 0;                  
    IntJ daqJsnRetCodeMask = 0;                 
    IntJ daqJsnHLTErrorEvents = 0;                  

    //write out HLT metadata jsn
    Json::Value hltDaqJsn;
    hltDaqJsn[DataPoint::SOURCE] = sourceHost;
    hltDaqJsn[DataPoint::DEFINITION] = sOutDef.str();

    hltDaqJsn[DataPoint::DATA].append((unsigned int)daqJsnProcessed.value());
    hltDaqJsn[DataPoint::DATA].append((unsigned int)daqJsnAccepted.value());
    hltDaqJsn[DataPoint::DATA].append((unsigned int)daqJsnErrorEvents.value());
    hltDaqJsn[DataPoint::DATA].append((unsigned int)daqJsnRetCodeMask.value());
    hltDaqJsn[DataPoint::DATA].append(hltJsnFilelist.value());
    hltDaqJsn[DataPoint::DATA].append((unsigned int)hltJsnFilesize.value());
    hltDaqJsn[DataPoint::DATA].append(hltJsnInputFiles.value());
    hltDaqJsn[DataPoint::DATA].append(hltJsnFileAdler32);
    hltDaqJsn[DataPoint::DATA].append(iSummary->streamHLTDestination);
    hltDaqJsn[DataPoint::DATA].append((unsigned int)daqJsnHLTErrorEvents.value());

    result = writer.write(hltDaqJsn);

    std::stringstream ssHltDaqJsn;
    ssHltDaqJsn <<  "run" << std::setfill('0') << std::setw(6) << iRun << "_ls" << std::setfill('0') << std::setw(4) << iLs;
    ssHltDaqJsn << "_streamHLTRates_pid" << std::setfill('0') << std::setw(5) << getpid() << ".jsn";

    std::ofstream outHltDaqJsn( monPath + ssHltDaqJsn.str() );
    outHltDaqJsn<<result;
    outHltDaqJsn.close();

    //write out L1 metadata jsn
    Json::Value l1DaqJsn;
    l1DaqJsn[DataPoint::SOURCE] = sourceHost;
    l1DaqJsn[DataPoint::DEFINITION] = sOutDef.str();

    l1DaqJsn[DataPoint::DATA].append((unsigned int)daqJsnProcessed.value());
    l1DaqJsn[DataPoint::DATA].append((unsigned int)daqJsnAccepted.value());
    l1DaqJsn[DataPoint::DATA].append((unsigned int)daqJsnErrorEvents.value());
    l1DaqJsn[DataPoint::DATA].append((unsigned int)daqJsnRetCodeMask.value());
    l1DaqJsn[DataPoint::DATA].append(l1JsnFilelist.value());
    l1DaqJsn[DataPoint::DATA].append((unsigned int)l1JsnFilesize.value());
    l1DaqJsn[DataPoint::DATA].append(l1JsnInputFiles.value());
    l1DaqJsn[DataPoint::DATA].append(l1JsnFileAdler32);
    l1DaqJsn[DataPoint::DATA].append(iSummary->streamL1Destination);
    l1DaqJsn[DataPoint::DATA].append((unsigned int)daqJsnHLTErrorEvents.value());

    result = writer.write(l1DaqJsn);

    std::stringstream ssL1DaqJsn;
    ssL1DaqJsn <<  "run" << std::setfill('0') << std::setw(6) << iRun << "_ls" << std::setfill('0') << std::setw(4) << iLs;
    ssL1DaqJsn << "_streamL1Rates_pid" << std::setfill('0') << std::setw(5) << getpid() << ".jsn";

    std::ofstream outL1DaqJsn( monPath + ssL1DaqJsn.str() );
    outL1DaqJsn<<result;
    outL1DaqJsn.close();
  }

  //Delete the individual HistoJ pointers   
  delete iSummary->processed;

  delete iSummary->hltWasRun;
  delete iSummary->hltL1s   ;
  delete iSummary->hltPre   ;
  delete iSummary->hltAccept;
  delete iSummary->hltReject;
  delete iSummary->hltErrors;

  delete iSummary->hltDatasets;

  delete iSummary->L1AlgoAccept;         
  delete iSummary->L1TechAccept;              
  delete iSummary->L1AlgoAcceptPhysics;         
  delete iSummary->L1TechAcceptPhysics;              
  delete iSummary->L1AlgoAcceptCalibration;         
  delete iSummary->L1TechAcceptCalibration;              
  delete iSummary->L1AlgoAcceptRandom;         
  delete iSummary->L1TechAcceptRandom;              
  delete iSummary->L1Global;                       

  //Note: Do not delete the iSummary pointer. The framework does something with it later on    
  //      and deleting it results in a segmentation fault.  

}//End globalEndLuminosityBlockSummary function     


void
TriggerJSONMonitoring::writeDefJson(std::string path){

  std::ofstream outfile( path );
  outfile << "{" << std::endl;
  outfile << "   \"data\" : [" << std::endl;
  outfile << "      {" ;
  outfile << " \"name\" : \"Processed\"," ;  //***  
  outfile << " \"type\" : \"integer\"," ;
  outfile << " \"operation\" : \"histo\"}," << std::endl;

  outfile << "      {" ;
  outfile << " \"name\" : \"Path-WasRun\"," ;
  outfile << " \"type\" : \"integer\"," ;
  outfile << " \"operation\" : \"histo\"}," << std::endl;

  outfile << "      {" ;
  outfile << " \"name\" : \"Path-AfterL1Seed\"," ;
  outfile << " \"type\" : \"integer\"," ;
  outfile << " \"operation\" : \"histo\"}," << std::endl;

  outfile << "      {" ;
  outfile << " \"name\" : \"Path-AfterPrescale\"," ;
  outfile << " \"type\" : \"integer\"," ;
  outfile << " \"operation\" : \"histo\"}," << std::endl;

  outfile << "      {" ;
  outfile << " \"name\" : \"Path-Accepted\"," ;
  outfile << " \"type\" : \"integer\"," ;
  outfile << " \"operation\" : \"histo\"}," << std::endl;

  outfile << "      {" ;
  outfile << " \"name\" : \"Path-Rejected\"," ;
  outfile << " \"type\" : \"integer\"," ;
  outfile << " \"operation\" : \"histo\"}," << std::endl;

  outfile << "      {" ;
  outfile << " \"name\" : \"Path-Errors\"," ;
  outfile << " \"type\" : \"integer\"," ;
  outfile << " \"operation\" : \"histo\"}," << std::endl;

  outfile << "      {" ;
  outfile << " \"name\" : \"Dataset-Accepted\"," ;
  outfile << " \"type\" : \"integer\"," ;
  outfile << " \"operation\" : \"histo\"}," << std::endl;

  outfile << "      {" ;
  outfile << " \"name\" : \"Prescale-Index\"," ;
  outfile << " \"type\" : \"integer\"," ;
  outfile << " \"operation\" : \"sample\"}" << std::endl;

  outfile << "   ]" << std::endl;
  outfile << "}" << std::endl;

  outfile.close();
}//End writeDefJson function                    


void
TriggerJSONMonitoring::writeL1DefJson(std::string path){              

  std::ofstream outfile( path );
  outfile << "{" << std::endl;
  outfile << "   \"data\" : [" << std::endl;
  outfile << "      {" ;
  outfile << " \"name\" : \"Processed\"," ;
  outfile << " \"type\" : \"integer\"," ;
  outfile << " \"operation\" : \"histo\"}," << std::endl;

  outfile << "      {" ;
  outfile << " \"name\" : \"L1-AlgoAccepted\"," ;
  outfile << " \"type\" : \"integer\"," ;
  outfile << " \"operation\" : \"histo\"}," << std::endl;

  outfile << "      {" ;
  outfile << " \"name\" : \"L1-TechAccepted\"," ;
  outfile << " \"type\" : \"integer\"," ;
  outfile << " \"operation\" : \"histo\"}," << std::endl;

  outfile << "      {" ;
  outfile << " \"name\" : \"L1-AlgoAccepted-Physics\"," ;
  outfile << " \"type\" : \"integer\"," ;
  outfile << " \"operation\" : \"histo\"}," << std::endl;

  outfile << "      {" ;
  outfile << " \"name\" : \"L1-TechAccepted-Physics\"," ;
  outfile << " \"type\" : \"integer\"," ;
  outfile << " \"operation\" : \"histo\"}," << std::endl;

  outfile << "      {" ;
  outfile << " \"name\" : \"L1-AlgoAccepted-Calibration\"," ;
  outfile << " \"type\" : \"integer\"," ;
  outfile << " \"operation\" : \"histo\"}," << std::endl;

  outfile << "      {" ;
  outfile << " \"name\" : \"L1-TechAccepted-Calibration\"," ;
  outfile << " \"type\" : \"integer\"," ;
  outfile << " \"operation\" : \"histo\"}," << std::endl;

  outfile << "      {" ;
  outfile << " \"name\" : \"L1-AlgoAccepted-Random\"," ;
  outfile << " \"type\" : \"integer\"," ;
  outfile << " \"operation\" : \"histo\"}," << std::endl;

  outfile << "      {" ;
  outfile << " \"name\" : \"L1-TechAccepted-Random\"," ;
  outfile << " \"type\" : \"integer\"," ;
  outfile << " \"operation\" : \"histo\"}," << std::endl;

  outfile << "      {" ;
  outfile << " \"name\" : \"L1-Global\"," ;
  outfile << " \"type\" : \"integer\"," ;
  outfile << " \"operation\" : \"histo\"}" << std::endl;

  outfile << "   ]" << std::endl;
  outfile << "}" << std::endl;

  outfile.close();
}//End writeL1DefJson function            

