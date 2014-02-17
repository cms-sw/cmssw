/** \class HLTConfigProvider
 *
 * See header file for documentation
 *
 *  $Date: 2012/10/03 13:34:09 $
 *  $Revision: 1.68 $
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

static const bool useL1EventSetup(true);
static const bool useL1GtTriggerMenuLite(false);

// an empty dummy config data used when we fail to initialize 
static const HLTConfigData* s_dummyHLTConfigData()
{ static HLTConfigData dummyHLTConfigData;
  return &dummyHLTConfigData;
}

HLTConfigProvider::HLTConfigProvider():
  processName_(""),
  inited_(false),
  changed_(true),
  hltConfigData_(s_dummyHLTConfigData()),
  l1GtUtils_(new L1GtUtils())
{
  //  HLTConfigDataRegistry::instance()->extra().increment();
}

//HLTConfigProvider::~HLTConfigProvider() {
//  if (HLTConfigDataRegistry::instance()->extra().decrement()==0) {
//    HLTConfigDataRegistry::instance()->data().clear();
//  }
//}

HLTConfigProvider::HLTConfigCounterSentry::HLTConfigCounterSentry() {
  HLTConfigDataRegistry::instance()->extra().increment();
}

HLTConfigProvider::HLTConfigCounterSentry::HLTConfigCounterSentry(HLTConfigCounterSentry const&) {
  HLTConfigDataRegistry::instance()->extra().increment();
}

HLTConfigProvider::HLTConfigCounterSentry::HLTConfigCounterSentry(HLTConfigCounterSentry &&) {
  HLTConfigDataRegistry::instance()->extra().increment();
}

HLTConfigProvider::HLTConfigCounterSentry::~HLTConfigCounterSentry() {
  HLTConfigDataRegistry::instance()->extra().decrement();
}

bool HLTConfigProvider::init(const edm::Run& iRun, 
                             const edm::EventSetup& iSetup, 
                             const std::string& processName, 
                             bool& changed) {

   using namespace std;
   using namespace edm;

   LogInfo("HLTConfigData") << "Called (R) with processName '"
			    << processName
			    << "' for " << iRun.id() << endl;

   init(iRun.processHistory(),processName);

   /// L1 GTA V3: https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideL1TriggerL1GtUtils#Version_3
   l1GtUtils_->getL1GtRunCache(iRun,iSetup,useL1EventSetup,useL1GtTriggerMenuLite);

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
     LogInfo("HLTConfigData") << "Auto-discovered processName: '"
			      << processName_ << "'"
			      << endl;
   }
   if (processName_=="*") {
     clear();
     LogError("HLTConfigData") << "Auto-discovery of processName failed!"
			       << endl;
     return;
   }

   /// Check uniqueness (uniqueness should [soon] be enforced by Fw)
   unsigned int n(0);
   for (ProcessHistory::const_iterator hi=hb; hi!=he; ++hi) {
     if (hi->processName()==processName_) {n++;}
   }
   if (n>1) {
     clear();
     LogError("HLTConfigProvider") << " ProcessName '"<< processName_
				   << " found " << n
				   << " times in history!" << endl;
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
     LogError("HLTConfigProvider") << "Falling back to processName-only init!";
     clear();
     init(processName_);
     if (!inited_) {
       LogError("HLTConfigProvider") << "ProcessName not found in history!";
     }
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

   LogVerbatim("HLTConfigProvider") << "Unordered list of all process names found: "
				    << pNames << "." << endl;

   LogVerbatim("HLTConfigProvider") << "HLT TableName of each selected process: "
				    << hNames << "." << endl;

   if (nPSets==0) {
     clear();
     LogError("HLTConfigProvider") << " Process name '"
				   << processName
				   << "' not found in registry!" << endl;
     return;
   }
   if (psetID==ParameterSetID()) {
     clear();
     LogError("HLTConfigProvider") << " Process name '"
				   << processName
				   << "' found but ParameterSetID invalid!"
				   << endl;
     return;
   }
   if (nPSets>1) {
     clear();
     LogError("HLTConfigProvider") << " Process name '"
				   << processName
				   << " found " << nPSets
				   << " times in registry!" << endl;
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
   *l1GtUtils_    = L1GtUtils();

   return;
}


int HLTConfigProvider::prescaleSet(const edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  // return hltPrescaleTable_.set();
  l1GtUtils_->getL1GtRunCache(iEvent,iSetup,useL1EventSetup,useL1GtTriggerMenuLite);
  int errorTech(0);
  const int psfsiTech(l1GtUtils_->prescaleFactorSetIndex(iEvent,L1GtUtils::TechnicalTrigger,errorTech));
  int errorPhys(0);
  const int psfsiPhys(l1GtUtils_->prescaleFactorSetIndex(iEvent,L1GtUtils::AlgorithmTrigger,errorPhys));
  assert(psfsiTech==psfsiPhys);
  if ( (errorTech==0) && (errorPhys==0) &&
       (psfsiTech>=0) && (psfsiPhys>=0) && (psfsiTech==psfsiPhys) ) {
    return psfsiPhys;
  } else {
    /// error - notify user!
    edm::LogError("HLTConfigData")
      << " Error in determining HLT prescale set index from L1 data using L1GtUtils: "
      << " Tech/Phys error = " << errorTech << "/" << errorPhys
      << " Tech/Phys psfsi = " << psfsiTech << "/" << psfsiPhys;
    return -1;
  }
}

unsigned int HLTConfigProvider::prescaleValue(const edm::Event& iEvent, const edm::EventSetup& iSetup, const std::string& trigger) const {
  const int set(prescaleSet(iEvent,iSetup));
  if (set<0) {
    return 1;
  } else {
    return prescaleValue(static_cast<unsigned int>(set),trigger);
  }
}

std::pair<int,int>  HLTConfigProvider::prescaleValues(const edm::Event& iEvent, const edm::EventSetup& iSetup, const std::string& trigger) const {

  // start with setting both L1T and HLT prescale values to 0
  std::pair<int,int> result(std::pair<int,int>(0,0));

  // get HLT prescale (possible if HLT prescale set index is correctly found)
  const int set(prescaleSet(iEvent,iSetup));
  if (set<0) {
    result.second = -1;
  } else {
    result.second = static_cast<int>(prescaleValue(static_cast<unsigned int>(set),trigger));
  }

  // get L1T prescale - works only for those hlt trigger paths with
  // exactly one L1GT seed module which has exactly one L1T name as seed
  const unsigned int nL1GTSeedModules(hltL1GTSeeds(trigger).size());
  if (nL1GTSeedModules==0) {
    // no L1 seed module on path hence no L1 seed hence formally no L1 prescale
    result.first=1;
  } else if (nL1GTSeedModules==1) {
    l1GtUtils_->getL1GtRunCache(iEvent,iSetup,useL1EventSetup,useL1GtTriggerMenuLite);
    const std::string l1tname(hltL1GTSeeds(trigger).at(0).second);
    int               l1error(0);
    result.first = l1GtUtils_->prescaleFactor(iEvent,l1tname,l1error);
    if (l1error!=0) {
      edm::LogError("HLTConfigData")
	<< " Error in determining L1T prescale for HLT path: '"	<< trigger
	<< "' with L1T seed: '" << l1tname
	<< "' using L1GtUtils: error code: " << l1error
	<< ". (Note: only a single L1T name, not a bit number, is allowed as seed for a proper determination of the L1T prescale!)";
      result.first = -1;
    }
  } else {
    /// error - can't handle properly multiple L1GTSeed modules
    std::string dump("'"+hltL1GTSeeds(trigger).at(0).second+"'");
    for (unsigned int i=1; i!=nL1GTSeedModules; ++i) {
      dump += " * '"+hltL1GTSeeds(trigger).at(i).second+"'";
    }
    edm::LogError("HLTConfigData")
      << " Error in determining L1T prescale for HLT path: '" << trigger
      << "' has multiple L1GTSeed modules, " << nL1GTSeedModules
      << ", with L1 seeds: " << dump
      << ". (Note: at most one L1GTSeed module is allowed for a proper determination of the L1T prescale!)";
    result.first = -1;
  }

  return result;
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
