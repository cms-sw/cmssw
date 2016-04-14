/** \class L1TriggerJSONMonitoring
 *  
 * See header file for documentation
 *
 * 
 *  \author Aram Avetisyan
 * 
 */

#include "HLTrigger/JSONMonitoring/interface/L1TriggerJSONMonitoring.h"

#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "EventFilter/Utilities/interface/FastMonitoringService.h"

#include <fstream>
using namespace jsoncollector;

L1TriggerJSONMonitoring::L1TriggerJSONMonitoring(const edm::ParameterSet& ps) :
  level1Results_(ps.getParameter<edm::InputTag>("L1Results")),   
  level1ResultsToken_(consumes<GlobalAlgBlkBxCollection>(level1Results_))             
{

                                                     
}

L1TriggerJSONMonitoring::~L1TriggerJSONMonitoring()
{
}

void
L1TriggerJSONMonitoring::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("L1Results",edm::InputTag("hltGtStage2Digis"));                
  descriptions.add("L1TMonitoring", desc);
}

void
L1TriggerJSONMonitoring::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  using namespace std;
  using namespace edm;

  processed_++;

  int ex = iEvent.experimentType();
  if      (ex == 1) L1Global_[0]++; 
  else if (ex == 2) L1Global_[1]++;
  else if (ex == 3) L1Global_[2]++;
  else{
    LogDebug("L1TriggerJSONMonitoring") << "Not Physics, Calibration or Random. experimentType = " << ex << std::endl;
  }   

  //Get hold of L1TResults 
  edm::Handle<GlobalAlgBlkBxCollection> l1tResults;
  if (not iEvent.getByToken(level1ResultsToken_, l1tResults) or (l1tResults->begin(0) == l1tResults->end(0))){
    LogDebug("L1TriggerJSONMonitoring") << "Could not get L1 trigger results" << std::endl;
    return;
  }
  
  //The GlobalAlgBlkBxCollection is a vector of vectors, but the second layer can only ever
  //have one entry since there can't be more than one collection per bunch crossing. The "0"
  //here means BX = 0, the "begin" is used to access the first and only element
  std::vector<GlobalAlgBlk>::const_iterator algBlk = l1tResults->begin(0);
  
  for (unsigned int i = 0; i < algBlk->maxPhysicsTriggers; i++){
    if (algBlk->getAlgoDecisionFinal(i)){
      L1AlgoAccept_[i]++;
      if (ex == 1) L1AlgoAcceptPhysics_[i]++;
      if (ex == 2) L1AlgoAcceptCalibration_[i]++;
      if (ex == 3) L1AlgoAcceptRandom_[i]++;
    }
  }  
  
  //Prescale index
  prescaleIndex_ = static_cast<unsigned int>(algBlk->getPreScColumn());
  
  //Check that the prescale index hasn't changed inside a lumi section
  unsigned int newLumi = (unsigned int) iEvent.eventAuxiliary().luminosityBlock();
  if (oldLumi == newLumi and prescaleIndex_ != oldPrescaleIndex){
    LogWarning("L1TriggerJSONMonitoring")<<"Prescale index has changed from "<<oldPrescaleIndex<<" to "<<prescaleIndex_<<" inside lumi section "<<newLumi;
  }
  oldLumi = newLumi;
  oldPrescaleIndex = prescaleIndex_;

}//End analyze function     

void
L1TriggerJSONMonitoring::resetLumi(){
  //Reset total number of events     
  processed_ = 0;

  //Reset per-path counters 
  //Reset L1 per-algo counters -     
  for (unsigned int i = 0; i < L1AlgoAccept_.size(); i++) {
    L1AlgoAccept_[i]            = 0;
    L1AlgoAcceptPhysics_[i]     = 0;
    L1AlgoAcceptCalibration_[i] = 0;
    L1AlgoAcceptRandom_[i]      = 0;
  }
  //Reset L1 global counters -      
  for (unsigned int i = 0; i < L1GlobalType_.size(); i++) {
    L1Global_[i] = 0;
  }

  //Prescale index
  prescaleIndex_ = 100;

}//End resetLumi function  

void
L1TriggerJSONMonitoring::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup)
{
  edm::ESHandle<L1TUtmTriggerMenu> l1GtMenu;
  iSetup.get<L1TUtmTriggerMenuRcd>().get(l1GtMenu);

  //Get the run directory from the EvFDaqDirector                
  if (edm::Service<evf::EvFDaqDirector>().isAvailable()) baseRunDir_ = edm::Service<evf::EvFDaqDirector>()->baseRunDir();
  else                                                   baseRunDir_ = ".";

  std::string monPath = baseRunDir_ + "/";

  //Need this to get at maximum number of triggers
  GlobalAlgBlk alg = GlobalAlgBlk();

  //Update trigger and dataset names, clear L1 names and counters   
  L1AlgoNames_.resize(alg.maxPhysicsTriggers);         
  for (unsigned int i = 0; i < L1AlgoNames_.size(); i++) {
    L1AlgoNames_.at(i) = "";
  }
  
  //Get L1 algorithm trigger names -      
  const L1TUtmTriggerMenu* m_l1GtMenu;
  const std::map<std::string, L1TUtmAlgorithm>* m_algorithmMap;

  m_l1GtMenu = l1GtMenu.product();
  m_algorithmMap = &(m_l1GtMenu->getAlgorithmMap());

  for (std::map<std::string, L1TUtmAlgorithm>::const_iterator itAlgo = m_algorithmMap->begin(); itAlgo != m_algorithmMap->end(); itAlgo++) {
    int bitNumber = (itAlgo->second).getIndex();
    L1AlgoNames_.at(bitNumber) = itAlgo->first;
  }
  
  L1GlobalType_.clear();   
  L1Global_.clear();     
  
  //Set the experimentType -          
  L1GlobalType_.push_back( "Physics" );
  L1GlobalType_.push_back( "Calibration" );
  L1GlobalType_.push_back( "Random" );

  const unsigned int la = L1AlgoNames_.size();       
  const unsigned int lg = L1GlobalType_.size();      
  
  //Resize per-path counters   
  L1AlgoAccept_.resize(la);         
  L1AlgoAcceptPhysics_.resize(la);         
  L1AlgoAcceptCalibration_.resize(la);         
  L1AlgoAcceptRandom_.resize(la);         
  
  L1Global_.resize(lg);                 
  
  resetLumi();

  //Write the once-per-run files if not already written
  //Eventually must rewrite this with proper multithreading (i.e. globalBeginRun)
  bool expected = false;
  if( runCache()->wroteFiles.compare_exchange_strong(expected, true) ){
    runCache()->wroteFiles = true;

    unsigned int nRun = iRun.run();
        
    //Create definition file for L1 Rates -  
    std::stringstream ssL1Jsd;
    ssL1Jsd << "run" << std::setfill('0') << std::setw(6) << nRun << "_ls0000";
    ssL1Jsd << "_streamL1Rates_pid" << std::setfill('0') << std::setw(5) << getpid() << ".jsd";
    stL1Jsd_ = ssL1Jsd.str();

    writeL1DefJson(baseRunDir_ + "/" + stL1Jsd_);
    
    //Write ini files    
    //L1
    Json::Value l1Ini;
    Json::StyledWriter writer;

    Json::Value l1AlgoNamesVal(Json::arrayValue);
    for (unsigned int ui = 0; ui < L1AlgoNames_.size(); ui++){
      l1AlgoNamesVal.append(L1AlgoNames_.at(ui));
    }

    Json::Value eventTypeVal(Json::arrayValue);
    for (unsigned int ui = 0; ui < L1GlobalType_.size(); ui++){
      eventTypeVal.append(L1GlobalType_.at(ui));
    }

    l1Ini["L1-Algo-Names"] = l1AlgoNamesVal;
    l1Ini["Event-Type"]    = eventTypeVal;
    
    std::string && result = writer.write(l1Ini);
  
    std::stringstream ssL1Ini;
    ssL1Ini << "run" << std::setfill('0') << std::setw(6) << nRun << "_ls0000_streamL1Rates_pid" << std::setfill('0') << std::setw(5) << getpid() << ".ini";
    
    std::ofstream outL1Ini( monPath + ssL1Ini.str() );
    outL1Ini<<result;
    outL1Ini.close();
  }

  //Initialize variables for verification of prescaleIndex
  oldLumi          = 0;
  oldPrescaleIndex = 100;

}//End beginRun function        

void L1TriggerJSONMonitoring::beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup& iSetup){ resetLumi(); }

std::shared_ptr<l1Json::lumiVars>
L1TriggerJSONMonitoring::globalBeginLuminosityBlockSummary(const edm::LuminosityBlock& iLumi, const edm::EventSetup& iSetup, const LuminosityBlockContext* iContext)
{
  std::shared_ptr<l1Json::lumiVars> iSummary(new l1Json::lumiVars);

  unsigned int MAXPATHS = 512;

  iSummary->processed = new HistoJ<unsigned int>(1, 1);

  iSummary->prescaleIndex = 100;

  iSummary->L1AlgoAccept            = new HistoJ<unsigned int>(1, MAXPATHS);                            
  iSummary->L1AlgoAcceptPhysics     = new HistoJ<unsigned int>(1, MAXPATHS);                            
  iSummary->L1AlgoAcceptCalibration = new HistoJ<unsigned int>(1, MAXPATHS);                            
  iSummary->L1AlgoAcceptRandom      = new HistoJ<unsigned int>(1, MAXPATHS);                            
  iSummary->L1Global                = new HistoJ<unsigned int>(1, MAXPATHS);  

  iSummary->baseRunDir           = "";
  iSummary->stL1Jsd              = "";
  iSummary->streamL1Destination  = "";

  return iSummary;
}//End globalBeginLuminosityBlockSummary function  

void
L1TriggerJSONMonitoring::endLuminosityBlockSummary(const edm::LuminosityBlock& iLumi, const edm::EventSetup& iEventSetup, l1Json::lumiVars* iSummary) const{

  //Whichever stream gets there first does the initialiazation 
  if (iSummary->L1AlgoAccept->value().size() == 0){
    iSummary->processed->update(processed_);

    iSummary->prescaleIndex = prescaleIndex_;

    iSummary->baseRunDir = baseRunDir_;
    
    for (unsigned int ui = 0; ui < L1AlgoAccept_.size(); ui++){    
      iSummary->L1AlgoAccept           ->update(L1AlgoAccept_.at(ui));
      iSummary->L1AlgoAcceptPhysics    ->update(L1AlgoAcceptPhysics_.at(ui));
      iSummary->L1AlgoAcceptCalibration->update(L1AlgoAcceptCalibration_.at(ui));
      iSummary->L1AlgoAcceptRandom     ->update(L1AlgoAcceptRandom_.at(ui));
    }
    for (unsigned int ui = 0; ui < L1GlobalType_.size(); ui++){    
      iSummary->L1Global    ->update(L1Global_.at(ui));
    }
    iSummary->stL1Jsd = stL1Jsd_;      

    iSummary->streamL1Destination  = runCache()->streamL1Destination;
  }

  else{
    iSummary->processed->value().at(0) += processed_;

    for (unsigned int ui = 0; ui < L1AlgoAccept_.size(); ui++){                             
      iSummary->L1AlgoAccept->value().at(ui)            += L1AlgoAccept_.at(ui);
      iSummary->L1AlgoAcceptPhysics->value().at(ui)     += L1AlgoAcceptPhysics_.at(ui);
      iSummary->L1AlgoAcceptCalibration->value().at(ui) += L1AlgoAcceptCalibration_.at(ui);
      iSummary->L1AlgoAcceptRandom->value().at(ui)      += L1AlgoAcceptRandom_.at(ui);
    }
    for (unsigned int ui = 0; ui < L1Global_.size(); ui++){                               
      iSummary->L1Global->value().at(ui) += L1Global_.at(ui);
    }

  }

}//End endLuminosityBlockSummary function                                             


void
L1TriggerJSONMonitoring::globalEndLuminosityBlockSummary(const edm::LuminosityBlock& iLumi, const edm::EventSetup& iSetup, const LuminosityBlockContext* iContext, l1Json::lumiVars* iSummary)
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
    //L1 .jsndata file
    Json::Value l1JsnData;
    l1JsnData[DataPoint::SOURCE] = sourceHost;
    l1JsnData[DataPoint::DEFINITION] = iSummary->stL1Jsd;

    l1JsnData[DataPoint::DATA].append(iSummary->processed->toJsonValue());
    l1JsnData[DataPoint::DATA].append(iSummary->L1AlgoAccept           ->toJsonValue());
    l1JsnData[DataPoint::DATA].append(iSummary->L1AlgoAcceptPhysics    ->toJsonValue());
    l1JsnData[DataPoint::DATA].append(iSummary->L1AlgoAcceptCalibration->toJsonValue());
    l1JsnData[DataPoint::DATA].append(iSummary->L1AlgoAcceptRandom     ->toJsonValue());
    l1JsnData[DataPoint::DATA].append(iSummary->L1Global               ->toJsonValue());      

    l1JsnData[DataPoint::DATA].append(iSummary->prescaleIndex);

    std::string && result = writer.write(l1JsnData);

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

  delete iSummary->L1AlgoAccept;         
  delete iSummary->L1AlgoAcceptPhysics;         
  delete iSummary->L1AlgoAcceptCalibration;         
  delete iSummary->L1AlgoAcceptRandom;         
  delete iSummary->L1Global;                       

  //Note: Do not delete the iSummary pointer. The framework does something with it later on    
  //      and deleting it results in a segmentation fault.  

}//End globalEndLuminosityBlockSummary function     

void
L1TriggerJSONMonitoring::writeL1DefJson(std::string path){              

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
  outfile << " \"name\" : \"L1-AlgoAccepted-Physics\"," ;
  outfile << " \"type\" : \"integer\"," ;
  outfile << " \"operation\" : \"histo\"}," << std::endl;

  outfile << "      {" ;
  outfile << " \"name\" : \"L1-AlgoAccepted-Calibration\"," ;
  outfile << " \"type\" : \"integer\"," ;
  outfile << " \"operation\" : \"histo\"}," << std::endl;

  outfile << "      {" ;
  outfile << " \"name\" : \"L1-AlgoAccepted-Random\"," ;
  outfile << " \"type\" : \"integer\"," ;
  outfile << " \"operation\" : \"histo\"}," << std::endl;

  outfile << "      {" ;
  outfile << " \"name\" : \"L1-Global\"," ;
  outfile << " \"type\" : \"integer\"," ;
  outfile << " \"operation\" : \"histo\"}," << std::endl;

  outfile << "      {" ;
  outfile << " \"name\" : \"Prescale-Index\"," ;
  outfile << " \"type\" : \"integer\"," ;
  outfile << " \"operation\" : \"sample\"}" << std::endl;

  outfile << "   ]" << std::endl;
  outfile << "}" << std::endl;

  outfile.close();
}//End writeL1DefJson function            

