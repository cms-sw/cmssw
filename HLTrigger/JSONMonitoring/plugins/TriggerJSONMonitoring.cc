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

#include <fstream>

TriggerJSONMonitoring::TriggerJSONMonitoring(const edm::ParameterSet& ps)
{
  if (ps.exists("triggerResults")) triggerResults_ = ps.getParameter<edm::InputTag> ("triggerResults");
  else                             triggerResults_ = edm::InputTag("TriggerResults","","HLT");

  triggerResultsToken_ = consumes<edm::TriggerResults>(triggerResults_);

  if (ps.exists("L1GtObjectMapTag")) m_l1GtObjectMapTag = ps.getParameter<edm::InputTag>("L1GtObjectMapTag"); //DS                                 
  else                               m_l1GtObjectMapTag = edm::InputTag("hltL1GtObjectMap");  //DS                                                 

  m_l1GtObjectMapToken = consumes<L1GlobalTriggerObjectMapRecord>(m_l1GtObjectMapTag); //DS                                                        
}

TriggerJSONMonitoring::~TriggerJSONMonitoring()
{
}

void
TriggerJSONMonitoring::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("triggerResults",edm::InputTag("TriggerResults","","HLT"));
  desc.add<edm::InputTag>("L1GtObjectMapTag",edm::InputTag("hltL1GtObjectMap"));              //DS                                                 
  descriptions.add("triggerJSONMonitoring", desc);
}

void
TriggerJSONMonitoring::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  using namespace std;
  using namespace edm;

  processed_++;

  int ex = iEvent.experimentType();
  if (ex==1) L1Global_[0]++;       //need to change to make sure a trigger fired??                                                                 
  else if (ex==2) L1Global_[1]++;
  else if (ex==3) L1Global_[2]++;
  else{
    std::cout << "Not Physics, Calibration or Random. experimentType = " << ex << std::endl; //can delete                                          
    LogDebug("TriggerJSONMonitoring") << "Not Physics, Calibration or Random. experimentType = " << ex << std::endl;
  }
  //got to here                                                                                                                                    

  // get L1GlobalTriggerObjectMapRecord                                                                                                            
  edm::Handle<L1GlobalTriggerObjectMapRecord> l1gtObjectMapRecord;  //DS                                                                           
  iEvent.getByToken(m_l1GtObjectMapToken, l1gtObjectMapRecord);     //DS                                                                           

  L1GlobalTriggerObjectMapRecord L1GTObjectMapRecord = *l1gtObjectMapRecord.product();

  // get ObjectMaps from ObjectMapRecord - //DS                                                                                                    
  const vector<L1GlobalTriggerObjectMap>& objMapVec = L1GTObjectMapRecord.gtObjectMap();
  for (vector<L1GlobalTriggerObjectMap>::const_iterator itMap = objMapVec.begin(); itMap != objMapVec.end(); ++itMap) {

    std::string algName = itMap->algoName();

    bool algResult = false;
    algResult = itMap->algoGtlResult();

    int count = 0;
    // add decision to counter for each L1 algorithm                                                                                               
    for (unsigned int i=0; i<L1Names_.size(); i++) {
      if (L1Names_[i] == algName){
        count++;
        if( algResult )  L1Accept_[i]++;
      }
    }
    if (count == 0) {
      LogDebug("TriggerJSONMonitoring") << "This algorithm not in list of L1Names " << algName << std::endl;
      L1Names_.push_back( algName );
      L1Accept_.push_back( algResult ? 1: 0 );
    }
    if (count > 1) LogDebug("TriggerJSONMonitoring" ) << "Algorithm repeated " << count << " times" << std::endl;

    /*count = 0;
    // add decision to counter for each L1 technical trigger - NOT SURE IF algoName() includes technical triggers!!!                               
    for (unsigned int i=0; i<L1TechNames_.size(); i++) {
      if (L1TechNames_[i] == algName){
        count++;
	if( algResult )  L1TechAccept_[i]++;
      }
    }
    if (count == 0) {
      std::cout << "This algorithm not in list of L1TechNames " << algName << std::endl; //can delete                                              
      LogDebug("TriggerJSONMonitoring") << "This algorithm not in list of L1TechNames " << algName << std::endl;
      L1TechNames_.push_back( algName );
      L1TechAccept_.push_back( algResult ? 1: 0 );
    }
    if (count > 1) LogDebug("TriggerJSONMonitoring" ) << "Algoithm repeated " << count << " times" << std::endl;
    if (count > 1) std::cout << "Algoithm repeated " << count << " times" << std::endl;  //can delete                     */
  }

  //Retrieve L1GtUtils                                                                                                        
  L1GtUtils l1GtUtils;
  l1GtUtils.retrieveL1EventSetup(iSetup);

  //Get technical trigger info from l1GtUtils                                                                                 
  for (CItAlgo algo = technicalMap.begin(); algo != technicalMap.end(); ++algo) {

    std::string techName = (algo->second).algoName();

    int iErrorCode = 0;
    bool techResult = false;
    techResult = l1GtUtils.decisionAfterMask(iEvent, techName, iErrorCode);

    int count = 0;
    for (unsigned int i=0; i<L1TechNames_.size(); i++) {
      if (L1TechNames_[i] == techName){
        count++;
        if( techResult )  L1TechAccept_[i]++;
      }
    }
    if (count == 0) std::cout << "This algorithm not in list of L1TechNames " << techName << std::endl; //can delete           
    if (count > 1)  std::cout << "Algoithm repeated " << count << " times" << std::endl;  //can delete                         
  }

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
}//End analyze function                                                                                                                            

void
TriggerJSONMonitoring::resetRun(bool changed){

  //Update trigger and dataset names, clear L1 names and counters                                                                                  
  if (changed){
    hltNames_        = hltConfig_.triggerNames();
    datasetNames_    = hltConfig_.datasetNames();
    datasetContents_ = hltConfig_.datasetContents();

    L1Names_.clear();   //DS                                                                                                                       
    L1Accept_.clear();  //DS                                                                                                                       
    //Get L1 algorithm trigger names - //DS                                                                                                        
    for (CItAlgo itAlgo = algorithmMap.begin(); itAlgo != algorithmMap.end(); itAlgo++) {
      L1Names_.push_back( itAlgo->first );
    }
    L1TechNames_.clear();  //DS                                                                                                                    
    L1TechAccept_.clear(); //DS                                                                                                                    
    //Get L1 technical trigger names - //DS                                                                                                        
    for (CItAlgo itAlgo = technicalMap.begin(); itAlgo != technicalMap.end(); itAlgo++) {
      L1TechNames_.push_back( itAlgo->first );
    }
    L1GlobalType_.clear();  //DS                                                                                                                   
    L1Global_.clear(); //DS                                                                                                                        
    //Set the experimentType - //DS                                                                                                                
    L1GlobalType_.push_back( "Physics" );
    L1GlobalType_.push_back( "Calibration" );
    L1GlobalType_.push_back( "Random" );
  }

  const unsigned int n  = hltNames_.size();
  const unsigned int d  = datasetNames_.size();
  const unsigned int ln = L1Names_.size();      //DS                                                                                               
  const unsigned int lt = L1TechNames_.size();  //DS                                                                                               
  const unsigned int lg = L1GlobalType_.size(); //DS                                                                                               

  if (changed) {
    //Resize per-path counters                                                                                                                    
    hltWasRun_.resize(n);
    hltL1s_.resize(n);
    hltPre_.resize(n);
    hltAccept_.resize(n);
    hltReject_.resize(n);
    hltErrors_.resize(n);
    L1Accept_.resize(ln);           //DS                                                                                                          
    L1TechAccept_.resize(lt);       //DS                                                                                                          
    L1Global_.resize(lg);           //DS                                                                                                          
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
    hltL1s_[i]   = 0;
    hltPre_[i]   = 0;
    hltAccept_[i] = 0;
    hltReject_[i] = 0;
    hltErrors_[i] = 0;
  }
  //Reset per-dataset counter                                                                                                                      
  for (unsigned int i = 0; i < hltDatasets_.size(); i++) {
    hltDatasets_[i] = 0;
  }
  //Reset L1 per-algo counters - //DS                                                                                                              
  for (unsigned int i = 0; i < L1Names_.size(); i++) {
    L1Accept_[i] = 0;
  }
  //Reset L1 per-tech counters - //DS                                                                                                              
  for (unsigned int i = 0; i < L1TechNames_.size(); i++) {
    L1TechAccept_[i] = 0;
  }
  //Reset L1 global counters - //DS                                                                                                                
  for (unsigned int i = 0; i < L1GlobalType_.size(); i++) {
    L1Global_[i] = 0;
  }

}//End resetLumi function                                                                                                                          

void
TriggerJSONMonitoring::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup)
{
  //Get the run directory from the EvFDaqDirector                                                                                                  
  if (edm::Service<evf::EvFDaqDirector>().isAvailable()) baseRunDir_ = edm::Service<evf::EvFDaqDirector>()->baseRunDir();
  else                                                   baseRunDir_ = ".";

  //Get/update the L1 trigger menu from the EventSetup                                                                                             
  edm::ESHandle<L1GtTriggerMenu> l1GtMenu;           //DS                                                                                          
  iSetup.get<L1GtTriggerMenuRcd>().get(l1GtMenu);    //DS                                                                                          
  m_l1GtMenu = l1GtMenu.product();                   //DS                                                                                          
  algorithmMap = m_l1GtMenu->gtAlgorithmMap();       //DS                                                                                          
  technicalMap = m_l1GtMenu->gtTechnicalTriggerMap();//DS                                                                                          

  //Initialize hltConfig_                                                                                                                          
  bool changed = true;
  if (hltConfig_.init(iRun, iSetup, triggerResults_.process(), changed)) resetRun(changed);
  else{
    LogDebug("TriggerJSONMonitoring") << "HLTConfigProvider initialization failed!" << std::endl;
    return;
  }

  unsigned int nRun = iRun.run();

  //Create definition file for HLT Rates                                                                                                               
  std::stringstream ssHltJsd;
  ssHltJsd << "run" << nRun << "_ls0000";
  ssHltJsd << "_streamHLTRates_pid" << std::setfill('0') << std::setw(5) << getpid() << ".jsd";
  stHltJsd_ = ssHltJsd.str();

  writeDefJson(baseRunDir_ + "/" + stHltJsd_);

  //Create definition file for L1 Rates - //DS                                                                                                     
  std::stringstream ssL1Jsd;
  ssL1Jsd << "run" << nRun << "_ls0000";
  ssL1Jsd << "_streamL1Rates_pid" << std::setfill('0') << std::setw(5) << getpid() << ".jsd";
  stL1Jsd_ = ssL1Jsd.str();

  writeL1DefJson(baseRunDir_ + "/" + stL1Jsd_);

}//End beginRun function                                                                                                                           

void TriggerJSONMonitoring::beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup& iSetup){ resetLumi(); }

std::shared_ptr<hltJson::lumiVars>
TriggerJSONMonitoring::globalBeginLuminosityBlockSummary(const edm::LuminosityBlock& iLumi, const edm::EventSetup& iSetup, const LuminosityBlockContext* iContext)
{
  std::shared_ptr<hltJson::lumiVars> iSummary(new hltJson::lumiVars);

  unsigned int MAXPATHS = 500;

  iSummary->processed = new HistoJ<unsigned int>(1, 1);

  iSummary->hltWasRun = new HistoJ<unsigned int>(1, MAXPATHS);
  iSummary->hltL1s    = new HistoJ<unsigned int>(1, MAXPATHS);
  iSummary->hltPre    = new HistoJ<unsigned int>(1, MAXPATHS);
  iSummary->hltAccept = new HistoJ<unsigned int>(1, MAXPATHS);
  iSummary->hltReject = new HistoJ<unsigned int>(1, MAXPATHS);
  iSummary->hltErrors = new HistoJ<unsigned int>(1, MAXPATHS);

  iSummary->hltDatasets = new HistoJ<unsigned int>(1, MAXPATHS);

  iSummary->hltNames     = new HistoJ<std::string>(1, MAXPATHS);
  iSummary->datasetNames = new HistoJ<std::string>(1, MAXPATHS);

  iSummary->L1Accept     = new HistoJ<unsigned int>(1, MAXPATHS); //DS                                                                                                     
  iSummary->L1Names      = new HistoJ<std::string>(1, MAXPATHS);  //DS                                                                                                     
  iSummary->L1TechAccept = new HistoJ<unsigned int>(1, MAXPATHS); //DS                                                                                                     
  iSummary->L1TechNames  = new HistoJ<std::string>(1, MAXPATHS);  //DS                                                                                                     
  iSummary->L1Global     = new HistoJ<unsigned int>(1, MAXPATHS); //DS                                                                                                     
  iSummary->L1GlobalType = new HistoJ<std::string>(1, MAXPATHS);  //DS                                                                                                     

  return iSummary;
}//End globalBeginLuminosityBlockSummary function                                                                                                                          

void
TriggerJSONMonitoring::endLuminosityBlockSummary(const edm::LuminosityBlock& iLumi, const edm::EventSetup& iEventSetup, hltJson::lumiVars* iSummary) const{

  //Whichever stream gets there first does the initialiazation                                                                                                             
  if (iSummary->hltNames->value().size() == 0){
    iSummary->processed->update(processed_);

    for (unsigned int ui = 0; ui < hltNames_.size(); ui++){
      iSummary->hltWasRun->update(hltWasRun_.at(ui));
      iSummary->hltL1s   ->update(hltL1s_   .at(ui));
      iSummary->hltPre   ->update(hltPre_   .at(ui));
      iSummary->hltAccept->update(hltAccept_.at(ui));
      iSummary->hltReject->update(hltReject_.at(ui));
      iSummary->hltErrors->update(hltErrors_.at(ui));

      iSummary->hltNames->update(hltNames_.at(ui));
    }
    for (unsigned int ui = 0; ui < datasetNames_.size(); ui++){
      iSummary->hltDatasets->update(hltDatasets_.at(ui));

      iSummary->datasetNames->update(datasetNames_.at(ui));
    }
    iSummary->stHltJsd   = stHltJsd_;
    iSummary->baseRunDir = baseRunDir_;

    for (unsigned int ui = 0; ui < L1Names_.size(); ui++){  //DS                                                                                                           
      iSummary->L1Accept->update(L1Accept_.at(ui));
      iSummary->L1Names ->update(L1Names_ .at(ui));
    }
    for (unsigned int ui = 0; ui < L1TechNames_.size(); ui++){  //DS                                                                                                       
      iSummary->L1TechAccept->update(L1TechAccept_.at(ui));
      iSummary->L1TechNames ->update(L1TechNames_ .at(ui));
    }
    for (unsigned int ui = 0; ui < L1GlobalType_.size(); ui++){  //DS                                                                                                      
      iSummary->L1Global    ->update(L1Global_.at(ui));
      iSummary->L1GlobalType->update(L1GlobalType_.at(ui));
    }
    iSummary->stL1Jsd = stL1Jsd_;    //DS                                                                                                            

  }
  else{
    iSummary->processed->value().at(0) += processed_;

    for (unsigned int ui = 0; ui < hltNames_.size(); ui++){
      iSummary->hltWasRun->value().at(ui) += hltWasRun_.at(ui);
      iSummary->hltL1s   ->value().at(ui) += hltL1s_   .at(ui);
      iSummary->hltPre   ->value().at(ui) += hltPre_   .at(ui);
      iSummary->hltAccept->value().at(ui) += hltAccept_.at(ui);
      iSummary->hltReject->value().at(ui) += hltReject_.at(ui);
      iSummary->hltErrors->value().at(ui) += hltErrors_.at(ui);
    }
    for (unsigned int ui = 0; ui < datasetNames_.size(); ui++){
      iSummary->hltDatasets->value().at(ui) += hltDatasets_.at(ui);
    }
    for (unsigned int ui = 0; ui < L1Names_.size(); ui++){  //DS                                                                                                           
      iSummary->L1Accept->value().at(ui) += L1Accept_.at(ui);
    }
    for (unsigned int ui = 0; ui < L1TechNames_.size(); ui++){  //DS                                                                                                       
      iSummary->L1TechAccept->value().at(ui) += L1TechAccept_.at(ui);
    }
    for (unsigned int ui = 0; ui < L1GlobalType_.size(); ui++){  //DS                                                                                                      
      iSummary->L1Global->value().at(ui) += L1Global_.at(ui);
    }

  }

}//End endLuminosityBlockSummary function                                                                                                                                  


void
TriggerJSONMonitoring::globalEndLuminosityBlockSummary(const edm::LuminosityBlock& iLumi, const edm::EventSetup& iSetup, const LuminosityBlockContext* iContext, hltJson::lumiVars* iSummary)
{
  Json::StyledWriter writer;

  char hostname[33];
  gethostname(hostname,32);
  std::string sourceHost(hostname);

  //Get the output directory                                                                                                                                           
  std::string monPath = iSummary->baseRunDir + "/";

  std::stringstream sOutDef;
  sOutDef << monPath << "output_" << getpid() << ".jsd";

  unsigned int iLs  = iLumi.luminosityBlock();
  unsigned int iRun = iLumi.run();

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

  std::string && result = writer.write(hltJsnData);

  std::stringstream ssHltJsnData;
  ssHltJsnData <<  "run" << iRun << "_ls" << std::setfill('0') << std::setw(4) << iLs;
  ssHltJsnData << "_streamHLTRates_pid" << std::setfill('0') << std::setw(5) << getpid() << ".jsndata";

  std::ofstream outHltJsnData( monPath + ssHltJsnData.str() );
  outHltJsnData<<result;
  outHltJsnData.close();

  //L1 .jsndata file
  Json::Value l1JsnData;
  l1JsnData[DataPoint::SOURCE] = sourceHost;
  l1JsnData[DataPoint::DEFINITION] = iSummary->stL1Jsd;

  l1JsnData[DataPoint::DATA].append(iSummary->processed->toJsonValue());
  l1JsnData[DataPoint::DATA].append(iSummary->L1Accept ->toJsonValue());
  l1JsnData[DataPoint::DATA].append(iSummary->L1TechAccept ->toJsonValue()); //DS                                                                                               
  l1JsnData[DataPoint::DATA].append(iSummary->L1Global ->toJsonValue());     //DS                                                                                               
  result = writer.write(l1JsnData);

  std::stringstream ssL1JsnData;
  ssL1JsnData << "run" << iRun << "_ls" << std::setfill('0') << std::setw(4) << iLs;
  ssL1JsnData << "_streamL1Rates_pid" << std::setfill('0') << std::setw(5) << getpid() << ".jsndata";

  std::ofstream outL1JsnData( monPath + "/" + ssL1JsnData.str() );
  outL1JsnData<<result;
  outL1JsnData.close();

  //HLT and L1 .ini files. They are only written once per run, but must be
  //at the end of a lumi section because it needs the path and dataset names 

  bool writeIni= ( iLs==1 ? true:false);
  if (edm::Service<evf::EvFDaqDirector>().isAvailable()) writeIni = !(edm::Service<evf::EvFDaqDirector>()->registerStreamProducer("TriggerJSONMonitoring"));
  if (writeIni) {
    //HLT
    Json::Value hltIni;

    Json::Value hltNamesVal(Json::arrayValue);
    for (unsigned int ui = 0; ui < iSummary->hltNames->value().size(); ui++){
      hltNamesVal.append(iSummary->hltNames->value().at(ui));
    }

    Json::Value datasetNamesVal(Json::arrayValue);
    for (unsigned int ui = 0; ui < iSummary->datasetNames->value().size(); ui++){
      datasetNamesVal.append(iSummary->datasetNames->value().at(ui));
    }

    hltIni["Path-Names"]    = hltNamesVal;
    hltIni["Dataset-Names"] = datasetNamesVal;
    
    result = writer.write(hltIni);
  
    std::stringstream ssHltIni;
    ssHltIni << "run" << iRun << "_ls0000_streamHLTRates_pid" << std::setfill('0') << std::setw(5) << getpid() << ".ini";
    
    std::ofstream outHltIni( monPath + ssHltIni.str() );
    outHltIni<<result;
    outHltIni.close();
    
    //L1
    Json::Value l1Ini;

    Json::Value l1AlgoNamesVal(Json::arrayValue);
    for (unsigned int ui = 0; ui < iSummary->L1Names->value().size(); ui++){
      l1AlgoNamesVal.append(iSummary->L1Names->value().at(ui));
    }

    Json::Value l1TechNamesVal(Json::arrayValue);
    for (unsigned int ui = 0; ui < iSummary->L1TechNames->value().size(); ui++){
      l1TechNamesVal.append(iSummary->L1TechNames->value().at(ui));
    }

    Json::Value eventTypeVal(Json::arrayValue);
    for (unsigned int ui = 0; ui < iSummary->L1GlobalType->value().size(); ui++){
      eventTypeVal.append(iSummary->L1GlobalType->value().at(ui));
    }

    l1Ini["L1-Algo-Names"] = l1AlgoNamesVal;
    l1Ini["L1-Tech-Names"] = l1TechNamesVal;
    l1Ini["Event-Type"]    = eventTypeVal;
    
    result = writer.write(l1Ini);
  
    std::stringstream ssL1Ini;
    ssL1Ini << "run" << iRun << "_ls0000_streamL1Rates_pid" << std::setfill('0') << std::setw(5) << getpid() << ".ini";
    
    std::ofstream outL1Ini( monPath + ssL1Ini.str() );
    outL1Ini<<result;
    outL1Ini.close();
  }

  //Create special DAQ JSON file for L1 and HLT rates pseudo-streams
  //Only three variables are different between the files: 
  //the file list, the file size and the Adler32 value
  IntJ daqJsnProcessed_   = iSummary->processed->value().at(0);
  IntJ daqJsnAccepted_    = daqJsnProcessed_;
  IntJ daqJsnErrorEvents_ = 0;                  
  IntJ daqJsnRetCodeMask_ = 0;                 

  struct stat st;

  //HLT
  StringJ hltJsnFilelist_;
  hltJsnFilelist_.update(ssHltJsnData.str());                 

  const char* cName = (monPath+ssHltJsnData.str()).c_str();
  stat(cName, &st);
  
  IntJ hltJsnFilesize_    = st.st_size;                    
  StringJ hltJsnInputFiles_;               
  hltJsnInputFiles_.update("");
  IntJ hltJsnFileAdler32_ = cms::Adler32(cName, st.st_size);        
  
  Json::Value hltDaqJsn;
  hltDaqJsn[DataPoint::SOURCE] = sourceHost;
  hltDaqJsn[DataPoint::DEFINITION] = sOutDef.str();

  hltDaqJsn[DataPoint::DATA].append((unsigned int)daqJsnProcessed_.value());
  hltDaqJsn[DataPoint::DATA].append((unsigned int)daqJsnAccepted_.value());
  hltDaqJsn[DataPoint::DATA].append((unsigned int)daqJsnErrorEvents_.value());
  hltDaqJsn[DataPoint::DATA].append((unsigned int)daqJsnRetCodeMask_.value());
  hltDaqJsn[DataPoint::DATA].append(hltJsnFilelist_.value());
  hltDaqJsn[DataPoint::DATA].append((unsigned int)hltJsnFilesize_.value());
  hltDaqJsn[DataPoint::DATA].append(hltJsnInputFiles_.value());
  hltDaqJsn[DataPoint::DATA].append((unsigned int)hltJsnFileAdler32_.value());

  result = writer.write(hltDaqJsn);

  std::stringstream ssHltDaqJsn;
  ssHltDaqJsn <<  "run" << iRun << "_ls" << std::setfill('0') << std::setw(4) << iLs;
  ssHltDaqJsn << "_streamHLTRates_pid" << std::setfill('0') << std::setw(5) << getpid() << ".jsn";

  std::ofstream outHltDaqJsn( monPath + ssHltDaqJsn.str() );
  outHltDaqJsn<<result;
  outHltDaqJsn.close();

  //L1
  StringJ l1JsnFilelist_;
  l1JsnFilelist_.update(ssL1JsnData.str());                 

  cName = (monPath+ssL1JsnData.str()).c_str();
  stat(cName, &st);
  
  IntJ l1JsnFilesize_    = st.st_size;                    
  StringJ l1JsnInputFiles_;               
  l1JsnInputFiles_.update("");
  IntJ l1JsnFileAdler32_ = cms::Adler32(cName, st.st_size);        
  
  Json::Value l1DaqJsn;
  l1DaqJsn[DataPoint::SOURCE] = sourceHost;
  l1DaqJsn[DataPoint::DEFINITION] = sOutDef.str();

  l1DaqJsn[DataPoint::DATA].append((unsigned int)daqJsnProcessed_.value());
  l1DaqJsn[DataPoint::DATA].append((unsigned int)daqJsnAccepted_.value());
  l1DaqJsn[DataPoint::DATA].append((unsigned int)daqJsnErrorEvents_.value());
  l1DaqJsn[DataPoint::DATA].append((unsigned int)daqJsnRetCodeMask_.value());
  l1DaqJsn[DataPoint::DATA].append(l1JsnFilelist_.value());
  l1DaqJsn[DataPoint::DATA].append((unsigned int)l1JsnFilesize_.value());
  l1DaqJsn[DataPoint::DATA].append(l1JsnInputFiles_.value());
  l1DaqJsn[DataPoint::DATA].append((unsigned int)l1JsnFileAdler32_.value());

  result = writer.write(l1DaqJsn);

  std::stringstream ssL1DaqJsn;
  ssL1DaqJsn <<  "run" << iRun << "_ls" << std::setfill('0') << std::setw(4) << iLs;
  ssL1DaqJsn << "_streamL1Rates_pid" << std::setfill('0') << std::setw(5) << getpid() << ".jsn";

  std::ofstream outL1DaqJsn( monPath + ssL1DaqJsn.str() );
  outL1DaqJsn<<result;
  outL1DaqJsn.close();

  //Delete the individual HistoJ pointers                                                                                                                                  
  delete iSummary->processed;

  delete iSummary->hltWasRun;
  delete iSummary->hltL1s   ;
  delete iSummary->hltPre   ;
  delete iSummary->hltAccept;
  delete iSummary->hltReject;
  delete iSummary->hltErrors;

  delete iSummary->hltDatasets;

  delete iSummary->hltNames;
  delete iSummary->datasetNames;

  delete iSummary->L1Accept;     //DS                                                                                                                                      
  delete iSummary->L1Names;      //DS                                                                                                                                      
  delete iSummary->L1TechAccept; //DS                                                                                                                                      
  delete iSummary->L1TechNames;  //DS                                                                                                                                      
  delete iSummary->L1Global;     //DS                                                                                                                                      
  delete iSummary->L1GlobalType; //DS                                                                                                                                      

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
  outfile << " \"operation\" : \"histo\"}" << std::endl;

  outfile << "   ]" << std::endl;
  outfile << "}" << std::endl;

  outfile.close();
}//End writeDefJson function                                                                                                                                               


void
TriggerJSONMonitoring::writeL1DefJson(std::string path){  //DS                                                                                                             

  std::ofstream outfile( path );
  outfile << "{" << std::endl;
  outfile << "   \"data\" : [" << std::endl;
  outfile << "      {" ;
  outfile << " \"name\" : \"Processed\"," ;
  outfile << " \"type\" : \"integer\"," ;
  outfile << " \"operation\" : \"histo\"}," << std::endl;

  outfile << "      {" ;
  outfile << " \"name\" : \"L1-Accepted\"," ;
  outfile << " \"type\" : \"integer\"," ;
  outfile << " \"operation\" : \"histo\"}," << std::endl;

  outfile << "      {" ;
  outfile << " \"name\" : \"L1-TechAccepted\"," ;
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

