/** \class HLTConfigProvider
 *
 * See header file for documentation
 *
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "HLTrigger/HLTcore/interface/HLTConfigDataRegistry.h"
#include "FWCore/Utilities/interface/RegexMatch.h"
#include "FWCore/Utilities/interface/ThreadSafeRegistry.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <boost/regex.hpp> 

// an empty dummy config data used when we fail to initialize 
static const HLTConfigData* s_dummyHLTConfigData()
{ static const HLTConfigData dummyHLTConfigData;
  return &dummyHLTConfigData;
}

HLTConfigProvider::HLTConfigProvider():
  processName_(""),
  inited_(false),
  changed_(true),
  hltConfigData_(s_dummyHLTConfigData())
{
}

bool HLTConfigProvider::init(const edm::Run& iRun, 
                             const edm::EventSetup& iSetup, 
                             const std::string& processName, 
                             bool& changed) {

   using namespace std;
   using namespace edm;

   LogInfo("HLTConfigProvider")
     << "Called (R) with processName '" << processName << "' for " << iRun.id() << endl;

   init(iRun.processHistory(),processName);

   processName_=processName;
   changed=changed_;
   return inited_;

}

void HLTConfigProvider::init(const edm::ProcessHistory& iHistory, const std::string& processName) {

   using namespace std;
   using namespace edm;

   const ProcessHistory::const_iterator hb(iHistory.begin());
   const ProcessHistory::const_iterator he(iHistory.end());

   ProcessConfiguration processConfiguration;
   const edm::ParameterSet* processPSet(0);

   processName_=processName;
   if (processName_=="*") {
     // auto-discovery of process name
     for (ProcessHistory::const_iterator hi=hb; hi!=he; ++hi) {
       if (iHistory.getConfigurationForProcess(hi->processName(),processConfiguration)) {
	 processPSet = edm::pset::Registry::instance()->getMapped(processConfiguration.parameterSetID());
	 if ((processPSet!=0) && (processPSet->exists("hltTriggerSummaryAOD"))) {
	   processName_=hi->processName();
	 }	 
       }
     }
     LogInfo("HLTConfigProvider")
       << "Auto-discovered processName: '" << processName_ << "'" << endl;
   }
   if (processName_=="*") {
     LogError("HLTConfigProvider")
       << "Auto-discovery of processName failed!" << endl;
     clear();
     return;
   }

   /// Check uniqueness (uniqueness should [soon] be enforced by Fw)
   unsigned int n(0);
   for (ProcessHistory::const_iterator hi=hb; hi!=he; ++hi) {
     if (hi->processName()==processName_) {n++;}
   }
   if (n>1) {
     LogError("HLTConfigProvider")
       << " ProcessName '"<< processName_ << " found " << n << " times in history!" << endl;
     clear();
     return;
   }

   ///
   if (iHistory.getConfigurationForProcess(processName_,processConfiguration)) {
     if ((hltConfigData_ !=s_dummyHLTConfigData()) && (processConfiguration.parameterSetID() == hltConfigData_->id())) {
       changed_ = false;
       inited_  = true;
       return;
     } else {
       getDataFrom(processConfiguration.parameterSetID());
     }
   } else {
     LogError("HLTConfigProvider")
       << "Falling back to ProcessName-only init using ProcessName '"<<processName_<<"' !";
     init(processName_);
     return;
   }
}

void HLTConfigProvider::getDataFrom(const edm::ParameterSetID& iID)
{
  //is it in our registry?
  HLTConfigDataRegistry* reg = HLTConfigDataRegistry::instance();
  const HLTConfigData* d = reg->getMapped(iID);
  if(0 != d) {
    changed_ = true;
    inited_  = true;
    hltConfigData_ = d;
  } else {
    const edm::ParameterSet* processPSet = 0;
    if ( 0 != (processPSet = edm::pset::Registry::instance()->getMapped(iID))) {
       if (not processPSet->id().isValid()) {
         clear();
         edm::LogError("HLTConfigProvider") << "ProcessPSet found is empty!";
         changed_ = true; 
         inited_  = false; 
         hltConfigData_ = s_dummyHLTConfigData();
         return; 
       } else { 
         clear(); 
         reg->insertMapped( HLTConfigData(processPSet));
         changed_ = true; 
         inited_  = true; 
         hltConfigData_ = reg->getMapped(processPSet->id());
         return;
       }
     } else {
       clear();
       edm::LogError("HLTConfigProvider") << "ProcessPSet not found in regsistry!";
       changed_ = true;
       inited_  = false;
       hltConfigData_ = s_dummyHLTConfigData();       
       return;
     }
  }
  return;
}

void HLTConfigProvider::init(const std::string& processName)
{
   using namespace std;
   using namespace edm;

   // Obtain ParameterSetID for requested process (with name
   // processName) from pset registry
   string pNames("");
   string hNames("");
   const ParameterSet*   pset = 0;
   ParameterSetID psetID;
   unsigned int   nPSets(0);
   const edm::pset::Registry * registry_(pset::Registry::instance());
   const edm::pset::Registry::const_iterator rb(registry_->begin());
   const edm::pset::Registry::const_iterator re(registry_->end());
   for (edm::pset::Registry::const_iterator i = rb; i != re; ++i) {
     if (i->second.exists("@process_name")) {
       const std::string pName(i->second.getParameter<string>("@process_name"));
       pNames += pName+" ";
       if ( pName == processName ) {
         psetID = i->first;
         nPSets++;
         if ((hltConfigData_ != s_dummyHLTConfigData()) && (hltConfigData_->id()==psetID)) {
           hNames += tableName();
         } else if ( 0 != (pset = registry_->getMapped(psetID))) {
           if (pset->exists("HLTConfigVersion")) {
             const ParameterSet& HLTPSet(pset->getParameterSet("HLTConfigVersion"));
             if (HLTPSet.exists("tableName")) {
               hNames += HLTPSet.getParameter<string>("tableName")+" ";
             }
           }
         }
       }
     }
   }

   LogVerbatim("HLTConfigProvider")
     << "Unordered list of all process names found: " << pNames << "." << endl;

   LogVerbatim("HLTConfigProvider")
     << "HLT TableName of each selected process: " << hNames << "." << endl;

   if (nPSets==0) {
     LogError("HLTConfigProvider")
       << " Process name '" << processName << "' not found in registry!" << endl;
     clear();
     return;
   }
   if (psetID==ParameterSetID()) {
     LogError("HLTConfigProvider")
       << " Process name '" << processName << "' found but ParameterSetID invalid!" << endl;
     clear();
     return;
   }
   if (nPSets>1) {
     LogError("HLTConfigProvider")
       << " Process name '" << processName << " found " << nPSets << " times in registry!" << endl;
     clear();
     return;
   }

   getDataFrom(psetID);

   return;

}

void HLTConfigProvider::clear()
{
   // clear all data members

   processName_   = "";
   inited_        = false;
   changed_       = true;
   hltConfigData_ = s_dummyHLTConfigData();

   return;
}



const std::vector<std::string> HLTConfigProvider::matched(const std::vector<std::string>& inputs, const std::string& pattern) {
  std::vector<std::string> matched;
  const boost::regex regexp(edm::glob2reg(pattern));
  const unsigned int n(inputs.size());
  for (unsigned int i=0; i<n; ++i) {
    const std::string& input(inputs[i]);
    if (boost::regex_match(input,regexp)) matched.push_back(input);
  }
  return matched;
}

const std::string HLTConfigProvider::removeVersion(const std::string& trigger) {
  const boost::regex regexp("_v[0-9]+$");
  return boost::regex_replace(trigger,regexp,"");
}

const std::vector<std::string> HLTConfigProvider::restoreVersion(const std::vector<std::string>& inputs, const std::string& trigger) {
  return matched(inputs,trigger+"_v[0-9]+$");
}
