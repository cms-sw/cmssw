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
}

TriggerJSONMonitoring::~TriggerJSONMonitoring()
{
}

void
TriggerJSONMonitoring::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("triggerResults",edm::InputTag("TriggerResults","","HLT"));
  descriptions.add("triggerJSONMonitoring", desc);
}

void
TriggerJSONMonitoring::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  using namespace std;
  using namespace edm;

  processed_++;

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

  //Update trigger and dataset names
  if (changed){
    hltNames_        = hltConfig_.triggerNames();
    datasetNames_    = hltConfig_.datasetNames();
    datasetContents_ = hltConfig_.datasetContents();
  }

  const unsigned int n = hltNames_.size();
  const unsigned int d = datasetNames_.size();
  
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
}//End resetLumi function

void  
TriggerJSONMonitoring::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup)
{  
  //Get the run directory from the EvFDaqDirector
  if (edm::Service<evf::EvFDaqDirector>().isAvailable()) baseRunDir_ = edm::Service<evf::EvFDaqDirector>()->baseRunDir();
  else                                                   baseRunDir_ = ".";

  //Create mon directory if it doesn't exist
  boost::filesystem::path monPath = baseRunDir_ + "/mon";
  if (not boost::filesystem::is_directory(monPath)) boost::filesystem::create_directories(monPath);

  //Initialize hltConfig_
  bool changed = true;
  if (hltConfig_.init(iRun, iSetup, triggerResults_.process(), changed)) resetRun(changed);
  else{ 
    LogDebug("TriggerJSONMonitoring") << "HLTConfigProvider initialization failed!" << std::endl;
    return;
  }

  unsigned int nRun = iRun.run();

  //Create definition file for Rates
  std::stringstream sjsdr;
  sjsdr << baseRunDir_ + "/mon/run" << nRun << "_ls0000";    
  sjsdr << "_HLTRates_pid" << std::setfill('0') << std::setw(5) << getpid() << ".jsd";
  jsonRateDefFile_ = sjsdr.str();
  
  writeDefJson(jsonRateDefFile_);

  //Create definition file for Legend
  std::stringstream sjsdl;
  sjsdl << baseRunDir_ << "/mon/run" << nRun << "_ls0000";
  sjsdl << "_HLTRatesLegend_pid" << std::setfill('0') << std::setw(5) << getpid() << ".jsd";
  jsonLegendDefFile_ = sjsdl.str();
  
  writeDefLegJson(jsonLegendDefFile_);
  
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
    iSummary->jsonRateDefFile   = jsonRateDefFile_;
    iSummary->jsonLegendDefFile = jsonLegendDefFile_;
    iSummary->baseRunDir        = baseRunDir_;
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
  }
}//End endLuminosityBlockSummary function


void  
TriggerJSONMonitoring::globalEndLuminosityBlockSummary(const edm::LuminosityBlock& iLumi, const edm::EventSetup& iSetup, const LuminosityBlockContext* iContext, hltJson::lumiVars* iSummary)
{
  Json::StyledWriter writer;

  char hostname[33];
  gethostname(hostname,32);
  std::string sourceHost(hostname);

  //Get the mon directory
  std::string monPath = iSummary->baseRunDir + "/mon";

  unsigned int iLs  = iLumi.luminosityBlock();
  unsigned int iRun = iLumi.run();

  Json::Value hltRates;
  hltRates[DataPoint::SOURCE] = sourceHost;
  hltRates[DataPoint::DEFINITION] = iSummary->jsonRateDefFile;

  hltRates[DataPoint::DATA].append(iSummary->processed->toString());
  hltRates[DataPoint::DATA].append(iSummary->hltWasRun->toString());
  hltRates[DataPoint::DATA].append(iSummary->hltL1s   ->toString());
  hltRates[DataPoint::DATA].append(iSummary->hltPre   ->toString());
  hltRates[DataPoint::DATA].append(iSummary->hltAccept->toString());
  hltRates[DataPoint::DATA].append(iSummary->hltReject->toString());
  hltRates[DataPoint::DATA].append(iSummary->hltErrors->toString());

  hltRates[DataPoint::DATA].append(iSummary->hltDatasets->toString());

  std::string && result = writer.write(hltRates);

  std::stringstream sjsnr;
  sjsnr << monPath << "/run" << iRun << "_ls" << std::setfill('0') << std::setw(4) << iLs;
  sjsnr << "_HLTRates_pid" << std::setfill('0') << std::setw(5) << getpid() << ".jsn";

  std::ofstream outJsnFileR( sjsnr.str() );
  outJsnFileR<<result;
  outJsnFileR.close();

  Json::Value hltNames;
  hltNames[DataPoint::SOURCE] = sourceHost;
  hltNames[DataPoint::DEFINITION] = iSummary->jsonLegendDefFile;

  hltNames[DataPoint::DATA].append(iSummary->hltNames->toString());
  hltNames[DataPoint::DATA].append(iSummary->datasetNames->toString());

  result = writer.write(hltNames);

  std::stringstream sjsnl;
  sjsnl << monPath << "/run" << iRun << "_ls" << std::setfill('0') << std::setw(4) << iLs;
  sjsnl << "_HLTRatesLegend_pid" << std::setfill('0') << std::setw(5) << getpid() << ".jsn";

  std::ofstream outJsnFileL( sjsnl.str() );
  outJsnFileL<<result;
  outJsnFileL.close();
  
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
   outfile << " type = \"integer\"," ;
   outfile << " \"operation\" : \"sum\"}," << std::endl;

   outfile << "      {" ;
   outfile << " \"name\" : \"Path-WasRun\"," ;
   outfile << " type = \"integer\"," ;
   outfile << " \"operation\" : \"sum\"}," << std::endl;

   outfile << "      {" ;
   outfile << " \"name\" : \"Path-AfterL1Seed\"," ;
   outfile << " type = \"integer\"," ;
   outfile << " \"operation\" : \"sum\"}," << std::endl;

   outfile << "      {" ;
   outfile << " \"name\" : \"Path-AfterPrescale\"," ;
   outfile << " type = \"integer\"," ;
   outfile << " \"operation\" : \"sum\"}," << std::endl;

   outfile << "      {" ;
   outfile << " \"name\" : \"Path-Accepted\"," ;
   outfile << " type = \"integer\"," ;
   outfile << " \"operation\" : \"sum\"}," << std::endl;
 
   outfile << "      {" ;
   outfile << " \"name\" : \"Path-Rejected\"," ;
   outfile << " type = \"integer\"," ;
   outfile << " \"operation\" : \"sum\"}," << std::endl;

   outfile << "      {" ;
   outfile << " \"name\" : \"Path-Errors\"," ;
   outfile << " type = \"integer\"," ;
   outfile << " \"operation\" : \"sum\"}," << std::endl;

   outfile << "      {" ;
   outfile << " \"name\" : \"Dataset-Accepted\"," ;
   outfile << " type = \"integer\"," ;
   outfile << " \"operation\" : \"sum\"}" << std::endl;
   
   outfile << "   ]" << std::endl;
   outfile << "}" << std::endl;
   
   outfile.close(); 
}//End writeDefJson function

void 
TriggerJSONMonitoring::writeDefLegJson(std::string path){

   std::ofstream legfile(path);
   legfile << "{" << std::endl;
   legfile << "   \"data\" : [" << std::endl;
   legfile << "      {" ;
   legfile << " \"name\" : \"Path-Names\"," ;
   legfile << " type = \"string\"," ;
   legfile << " \"operation\" : \"cat\"}," << std::endl;

   legfile << "      {" ;
   legfile << " \"name\" : \"Dataset-Names\"," ;
   legfile << " type = \"string\"," ;
   legfile << " \"operation\" : \"cat\"}" << std::endl;
   legfile << "   ]" << std::endl;
   legfile << "}" << std::endl;
   legfile.close();
}//End writeDefLegJson function
