/** \class HLTriggerJSONMonitoring
 *  
 * See header file for documentation
 *
 * 
 *  \author Aram Avetisyan
 * 
 */

#include "HLTrigger/JSONMonitoring/interface/HLTriggerJSONMonitoring.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "EventFilter/Utilities/interface/FastMonitoringService.h"

#include <fstream>
using namespace jsoncollector;

HLTriggerJSONMonitoring::HLTriggerJSONMonitoring(const edm::ParameterSet& ps) :
  triggerResults_(ps.getParameter<edm::InputTag>("triggerResults")),
  triggerResultsToken_(consumes<edm::TriggerResults>(triggerResults_))
{

                                                     
}

HLTriggerJSONMonitoring::~HLTriggerJSONMonitoring()
{
}

void
HLTriggerJSONMonitoring::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("triggerResults",edm::InputTag("TriggerResults","","HLT"));
  descriptions.add("HLTriggerJSONMonitoring", desc);
}

void
HLTriggerJSONMonitoring::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  using namespace std;
  using namespace edm;

  processed_++;
  
  //Get hold of TriggerResults  
  Handle<TriggerResults> HLTR;
  if (not iEvent.getByToken(triggerResultsToken_, HLTR) or not HLTR.isValid()){
    LogDebug("HLTriggerJSONMonitoring") << "HLT TriggerResults with label ["+triggerResults_.encode()+"] not found!" << std::endl;
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
HLTriggerJSONMonitoring::resetRun(bool changed){

  //Update trigger and dataset names, clear L1 names and counters   
  if (changed){
    hltNames_        = hltConfig_.triggerNames();
    datasetNames_    = hltConfig_.datasetNames();
    datasetContents_ = hltConfig_.datasetContents();
  }

  const unsigned int n  = hltNames_.size();
  const unsigned int d  = datasetNames_.size();

  if (changed) {
    //Resize per-path counters   
    hltWasRun_.resize(n);
    hltL1s_.resize(n);
    hltPre_.resize(n);
    hltAccept_.resize(n);
    hltReject_.resize(n);
    hltErrors_.resize(n);

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
	if (label == "HLTL1TSeed")
	  posL1s_[i] = j;
	else if (label == "HLTPrescaler")
	  posPre_[i] = j;
      }
    }
  }
  resetLumi();
}//End resetRun function                  

void
HLTriggerJSONMonitoring::resetLumi(){
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

}//End resetLumi function  

void
HLTriggerJSONMonitoring::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup)
{
  //Get the run directory from the EvFDaqDirector                
  if (edm::Service<evf::EvFDaqDirector>().isAvailable()) baseRunDir_ = edm::Service<evf::EvFDaqDirector>()->baseRunDir();
  else                                                   baseRunDir_ = ".";

  std::string monPath = baseRunDir_ + "/";

  //Initialize hltConfig_     
  bool changed = true;
  if (hltConfig_.init(iRun, iSetup, triggerResults_.process(), changed)) resetRun(changed);
  else{
    LogDebug("HLTriggerJSONMonitoring") << "HLTConfigProvider initialization failed!" << std::endl;
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

    writeHLTDefJson(baseRunDir_ + "/" + stHltJsd_);
        
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
  }

}//End beginRun function        

void HLTriggerJSONMonitoring::beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup& iSetup){ resetLumi(); }

std::shared_ptr<hltJson::lumiVars>
HLTriggerJSONMonitoring::globalBeginLuminosityBlockSummary(const edm::LuminosityBlock& iLumi, const edm::EventSetup& iSetup, const LuminosityBlockContext* iContext)
{
  std::shared_ptr<hltJson::lumiVars> iSummary(new hltJson::lumiVars);

  unsigned int MAXPATHS = 1000;

  iSummary->processed = new HistoJ<unsigned int>(1, 1);

  iSummary->hltWasRun = new HistoJ<unsigned int>(1, MAXPATHS);
  iSummary->hltL1s    = new HistoJ<unsigned int>(1, MAXPATHS);
  iSummary->hltPre    = new HistoJ<unsigned int>(1, MAXPATHS);
  iSummary->hltAccept = new HistoJ<unsigned int>(1, MAXPATHS);
  iSummary->hltReject = new HistoJ<unsigned int>(1, MAXPATHS);
  iSummary->hltErrors = new HistoJ<unsigned int>(1, MAXPATHS);

  iSummary->hltDatasets = new HistoJ<unsigned int>(1, MAXPATHS);

  iSummary->baseRunDir           = "";
  iSummary->stHltJsd             = "";
  iSummary->streamHLTDestination = "";
  iSummary->streamHLTMergeType  = "";

  return iSummary;
}//End globalBeginLuminosityBlockSummary function  

void
HLTriggerJSONMonitoring::endLuminosityBlockSummary(const edm::LuminosityBlock& iLumi, const edm::EventSetup& iEventSetup, hltJson::lumiVars* iSummary) const{

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

    iSummary->stHltJsd   = stHltJsd_;
    iSummary->baseRunDir = baseRunDir_;
    
    iSummary->streamHLTDestination = runCache()->streamHLTDestination;
    iSummary->streamHLTMergeType   = runCache()->streamHLTMergeType;
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
  }

}//End endLuminosityBlockSummary function                                             


void
HLTriggerJSONMonitoring::globalEndLuminosityBlockSummary(const edm::LuminosityBlock& iLumi, const edm::EventSetup& iSetup, const LuminosityBlockContext* iContext, hltJson::lumiVars* iSummary)
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
    hltDaqJsn[DataPoint::DATA].append(iSummary->streamHLTMergeType);
    hltDaqJsn[DataPoint::DATA].append((unsigned int)daqJsnHLTErrorEvents.value());

    result = writer.write(hltDaqJsn);

    std::stringstream ssHltDaqJsn;
    ssHltDaqJsn <<  "run" << std::setfill('0') << std::setw(6) << iRun << "_ls" << std::setfill('0') << std::setw(4) << iLs;
    ssHltDaqJsn << "_streamHLTRates_pid" << std::setfill('0') << std::setw(5) << getpid() << ".jsn";

    std::ofstream outHltDaqJsn( monPath + ssHltDaqJsn.str() );
    outHltDaqJsn<<result;
    outHltDaqJsn.close();
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

  //Note: Do not delete the iSummary pointer. The framework does something with it later on    
  //      and deleting it results in a segmentation fault.  

  //Reninitalize HistoJ pointers to nullptr   
  iSummary->processed   = nullptr;

  iSummary->hltWasRun   = nullptr;
  iSummary->hltL1s      = nullptr;
  iSummary->hltPre      = nullptr;
  iSummary->hltAccept   = nullptr;
  iSummary->hltReject   = nullptr;
  iSummary->hltErrors   = nullptr;

  iSummary->hltDatasets = nullptr;

}//End globalEndLuminosityBlockSummary function     


void
HLTriggerJSONMonitoring::writeHLTDefJson(std::string path){

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
}//End writeHLTDefJson function                    
