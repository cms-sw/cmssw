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

  processed_.value()++;

  //Get hold of TriggerResults
  Handle<TriggerResults> HLTR;
  iEvent.getByToken(triggerResultsToken_, HLTR);
  if (!HLTR.isValid()) {
    LogDebug("TriggerJSONMonitoring") << "HLT TriggerResults with label ["+triggerResults_.encode()+"] not found!" << std::endl;
    return;
  }
  
  // decision for each HLT path
  const unsigned int n(hltNames_.size());
  for (unsigned int i=0; i<n; i++) {
    if (HLTR->accept(i)){
      hltPaths_[i].value()++;
    }
  }

}

void 
TriggerJSONMonitoring::reset(bool changed ){

  processed_.value() = 0;

  // update trigger names
  if (changed) hltNames_ = hltConfig_.triggerNames();

  const unsigned int n = hltNames_.size();

  if (changed) {
    // resize per-path counter
    hltPaths_.resize(n);
  }

  // reset per-path counter
  for (unsigned int i = 0; i < n; i++) {
    hltPaths_[i].value() = 0;
  }

}


void 
TriggerJSONMonitoring::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup)
{

   //Get the run directory from the EvFDaqDirector
   baseRunDir = edm::Service<evf::EvFDaqDirector>()->baseRunDir();
  
   // initialize hltConfig_
   bool changed = true;
   if (hltConfig_.init(iRun, iSetup, triggerResults_.process(), changed)) {
     reset(true);
   }

   // set up JSON
   processed_.setName("Processed");

   DataPointDefinition outJsonTemp_;

   outJsonTemp_.setDefaultGroup("data");
   outJsonTemp_.addLegendItem("Processed","integer",DataPointDefinition::SUM);

   const unsigned int n = hltNames_.size();

   for(unsigned int i =0; i<n; i++){
     hltPaths_[i].setName( hltNames_[i] );
     outJsonTemp_.addLegendItem(hltNames_[i],"integer",DataPointDefinition::SUM);
   }

   outJson_ = outJsonTemp_;

   // create JSON definition file
   unsigned int nRun = iRun.run();

   std::stringstream sjsd;
   sjsd << baseRunDir << "/mon/HLTRates_run"<< nRun; 
   sjsd << "_pid" << std::setfill('0') << std::setw(5) << getpid() << ".jsd";
   jsonDefinitionFile = sjsd.str();
   
   std::ofstream outfile( jsonDefinitionFile );
   outfile << "{" << std::endl;
   outfile << "   \"data\" : [" << std::endl;
   outfile << "      {" << std::endl;
   outfile << "         \"name\" : \"Processed\"," << std::endl;
   outfile << "         \"operation\" : \"sum\"," << std::endl;
   outfile << "         \"type\" : \"integer\"" << std::endl;
   
   for(unsigned int j =0; j<n; j++){
     outfile << "      }," << std::endl;
     outfile << "      {" << std::endl;
     outfile << "         \"name\" : \"" << hltNames_[j] << "\"," << std::endl;
     outfile << "         \"operation\" : \"sum\"," << std::endl;
     outfile << "         \"type\" : \"integer\"" << std::endl;
   } 
   
   outfile << "      }" << std::endl;
   outfile << "   ]" << std::endl;
   outfile << "}" << std::endl;
   
   outfile.close();

   //std::cout << "CurrentRunDir is " << RunDir << std::endl;  //***
}

void 
TriggerJSONMonitoring::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{

  std::string outJsonDefName = jsonDefinitionFile;

  jsonMonitor_.reset(new jsoncollector::FastMonitor(&outJson_, false));
  jsonMonitor_->setDefPath(outJsonDefName);
  jsonMonitor_->registerGlobalMonitorable(&processed_,false);

  for(unsigned int i = 0; i < hltPaths_.size(); i++){
    jsonMonitor_->registerGlobalMonitorable(&hltPaths_[i], false);
  }

  jsonMonitor_->commit(nullptr);

  reset();
}

void 
TriggerJSONMonitoring::endLuminosityBlock(edm::LuminosityBlock const& ls, edm::EventSetup const&)
{

  unsigned int uiLS  = ls.luminosityBlock();
  unsigned int uiRun = ls.run();

  jsonMonitor_->snap(uiLS);

  std::stringstream ss;
  ss << baseRunDir << "/mon/" << "HLTRates";
  ss << "_run" << uiRun << "_ls" << std::setfill('0') << std::setw(4) << uiLS;
  ss << "_pid" << std::setfill('0') << std::setw(5) << getpid() << ".jsn";
  std::string outputJsonNameStream = ss.str();
  
  jsonMonitor_->outputFullJSON(outputJsonNameStream, uiLS);

  // Debug output
  LogDebug("TriggerJSONMonitoring") << "i hltNames #Accepted" << std::endl;
  const unsigned int size = hltNames_.size();
  for (unsigned int i = 0; i < size; i++) {
    LogDebug("TriggerJSONMonitoring") << i                    << " "
				      << hltNames_[i]         << " "
				      << hltPaths_[i].value() << std::endl;
  }

}
