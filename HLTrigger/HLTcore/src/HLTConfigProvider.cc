/** \class HLTConfigProvider
 *
 * See header file for documentation
 *
 *  $Date: 2010/03/16 06:40:41 $
 *  $Revision: 1.42 $
 *
 *  \author Martin Grunewald
 *
 */

#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtUtils.h"

#include <algorithm>
#include <iostream>

void HLTConfigProvider::clear()
{
   using namespace std;
   using namespace edm;
   using namespace trigger;

   // clear data members

   processName_ = "";
   processPSet_ = ParameterSet();

   tableName_   = "/dev/null";

   triggerNames_.clear();
   moduleLabels_.clear();

   triggerIndex_.clear();
   moduleIndex_.clear();

   hltL1GTSeeds_.clear();

   streamNames_.clear();
   streamContents_.clear();
   streamIndex_.clear();

   datasetNames_.clear();
   datasetContents_.clear();
   datasetIndex_.clear();

   hltPrescaleTable_ = HLTPrescaleTable();

   return;
}

/*
bool HLTConfigProvider::init(const edm::Event& iEvent, const std::string& processName, bool& changed)
{
   using namespace std;
   using namespace edm;

   LogInfo("HLTConfigProvider") << "Called (E) with processName '"
				<< processName << "'." << endl;

   ParameterSet eventPSet;
   if (iEvent.getProcessParameterSet(processName,eventPSet)) {
     if ( processPSet_==eventPSet ) {
       changed=false;
     } else { 
       clear();
       processName_  =processName;
       processPSet_  =eventPSet;
       extract();
       changed=true;
     }
     return true;
   } else {
     clear();
     LogError("HLTConfigProvider")
       << "Event ProcessPSet not found for processName '"
       << processName <<"'!";
     changed=true;
     return false;
   }

}
*/

bool HLTConfigProvider::init(const edm::ProcessHistory& iHistory, const edm::EventSetup& iSetup, const std::string& processName, bool& changed) {
  return init(iHistory,processName,changed);
}

bool HLTConfigProvider::init(const edm::ProcessHistory& iHistory, const std::string& processName, bool& changed) {

   using namespace std;
   using namespace edm;

   /// Check uniqueness (uniqueness should [soon] be enforced by Fw)
   const ProcessHistory::const_iterator hb(iHistory.begin());
   const ProcessHistory::const_iterator he(iHistory.end());
   unsigned int n(0);
   for (ProcessHistory::const_iterator hi=hb; hi!=he; ++hi) {
     if (hi->processName()==processName) {n++;}
   }
   if (n>1) {
     clear();
     LogError("HLTConfigProvider") << " ProcessName '"<< processName
				   << " found " << n
				   << " times in history!" << endl;
     changed=true;
     return false;
   }

   ///
   ProcessConfiguration processConfiguration;
   if (iHistory.getConfigurationForProcess(processName,processConfiguration)) {
     ParameterSet processPSet;
     if ((processPSet_!=ParameterSet()) && (processConfiguration.parameterSetID() == processPSet_.id())) {
       changed=false;
       return true;
     } else if (pset::Registry::instance()->getMapped(processConfiguration.parameterSetID(),processPSet)) {
       if (processPSet==ParameterSet()) {
	 clear();
	 LogError("HLTConfigProvider") << "ProcessPSet found is empty!";
	 changed=true;
	 return false;
       } else {
	 clear();
	 processName_=processName;
	 processPSet_=processPSet;
	 extract();
	 changed=true;
	 return true;
       }
     } else {
       clear();
       LogError("HLTConfigProvider") << "ProcessPSet not found in regsistry!";
       changed=true;
       return false;
     }
   } else {
     clear();
     changed=true;
     if (init(processName)) {
       return true;
     } else {
       LogError("HLTConfigProvider") << "ProcessName not found in history!";
       return false;
     }
   }
}

bool HLTConfigProvider::init(const edm::LuminosityBlock& iLumiBlock, const edm::EventSetup& iSetup, const std::string& processName, bool& changed) {
  return init(iLumiBlock,processName,changed);
}

bool HLTConfigProvider::init(const edm::LuminosityBlock& iLumiBlock, const std::string& processName, bool& changed) {
   using namespace std;
   using namespace edm;

   LogInfo("HLTConfigProvider") << "Called (L) with processName '"
				<< processName << "'." << endl;

   const ProcessHistory& processHistory(iLumiBlock.processHistory());
   return init(processHistory,processName,changed);
}

bool HLTConfigProvider::init(const edm::Run& iRun, const edm::EventSetup& iSetup, const std::string& processName, bool& changed) {
  return init(iRun,processName,changed);
}

bool HLTConfigProvider::init(const edm::Run& iRun, const std::string& processName, bool& changed) {
   using namespace std;
   using namespace edm;

   LogInfo("HLTConfigProvider") << "Called (R) with processName '"
				<< processName << "'." << endl;

   const ProcessHistory& processHistory(iRun.processHistory());
   return init(processHistory,processName,changed);
}

bool HLTConfigProvider::init(const edm::Event& iEvent, const edm::EventSetup& iSetup, const std::string& processName, bool& changed) {
  return init(iEvent,processName,changed);
}

bool HLTConfigProvider::init(const edm::Event& iEvent, const std::string& processName, bool& changed) {
   using namespace std;
   using namespace edm;

   LogInfo("HLTConfigProvider") << "Called (E) with processName '"
				<< processName << "'." << endl;

   const ProcessHistory& processHistory(iEvent.processHistory());
   return init(processHistory,processName,changed);
}

bool HLTConfigProvider::init(const std::string& processName)
{
   using namespace std;
   using namespace edm;

   LogInfo("HLTConfigProvider") << "Called (N) with processName '"
				<< processName << "'." << endl;

   LogVerbatim("HLTConfigProvider")
     << "This 1-parameter init method fails (returns false) when processing "
     << "file(s) containing events accepted by different HLT tables - "
     << "for such cases use the 3-parameter init method called each event!"
     << endl;

   // Obtain ParameterSetID for requested process (with name
   // processName) from pset registry
   string pNames("");
   string hNames("");
   ParameterSet   pset;
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
	 if ((processPSet_!=ParameterSet()) && (processPSet_.id()==psetID)) {
	   hNames += tableName();
	 } else if (registry_->getMapped(psetID,pset)) {
	   if (pset.exists("HLTConfigVersion")) {
	     const ParameterSet HLTPSet(pset.getParameter<ParameterSet>("HLTConfigVersion"));
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
     return false;
   }
   if (psetID==ParameterSetID()) {
     clear();
     LogError("HLTConfigProvider") << " Process name '"
				   << processName
				   << "' found but ParameterSetID invalid!"
				   << endl;
     return false;
   }
   if (nPSets>1) {
     clear();
     LogError("HLTConfigProvider") << " Process name '"
				   << processName
				   << " found " << nPSets
				   << " times in registry!" << endl;
     return false;
   }

   // Obtain ParameterSet from ParameterSetID
   if (!(registry_->getMapped(psetID,pset))) {
     clear();
     LogError("HLTConfigProvider") << " ProcessPSet for ProcessPSetID '"
				   << psetID
				   << "' not found in registry!" << endl;
     return false;
   }

   if ((processName_!=processName) || (processPSet_!=pset)) {
     clear();
     processName_=processName;
     processPSet_=pset;
     extract();
   }

   return true;

}

void HLTConfigProvider::extract()
{
   using namespace std;
   using namespace edm;
   using namespace trigger;

   // Obtain PSet containing table name (available only in 2_1_10++ files)
   if (processPSet_.exists("HLTConfigVersion")) {
     const ParameterSet HLTPSet(processPSet_.getParameter<ParameterSet>("HLTConfigVersion"));
     if (HLTPSet.exists("tableName")) {
       tableName_=HLTPSet.getParameter<string>("tableName");
     }
   }
   LogVerbatim("HLTConfigProvider") << "ProcessPSet with HLT: "
				    << tableName();

   // Extract trigger paths (= paths - end_paths)
   triggerNames_= processPSet_.getParameter<ParameterSet>("@trigger_paths").getParameter<vector<string> >("@trigger_paths");

   // Obtain module labels of all modules on all trigger paths
   const unsigned int n(size());
   moduleLabels_.reserve(n);
   for (unsigned int i=0;i!=n; ++i) {
     moduleLabels_.push_back(processPSet_.getParameter<vector<string> >(triggerNames_[i]));
   }

   // Fill index maps for fast lookup
   moduleIndex_.resize(n);
   for (unsigned int i=0; i!=n; ++i) {
     triggerIndex_[triggerNames_[i]]=i;
     moduleIndex_[i].clear();
     const unsigned int m(size(i));
     for (unsigned int j=0; j!=m; ++j) {
       moduleIndex_[i][moduleLabels_[i][j]]=j;
     }
   }

   // Extract and fill HLTLevel1GTSeed information for each trigger path
   hltL1GTSeeds_.resize(n);
   for (unsigned int i=0; i!=n; ++i) {
     hltL1GTSeeds_[i].clear();
     const unsigned int m(size(i));
     for (unsigned int j=0; j!=m; ++j) {
       const string& label(moduleLabels_[i][j]);
       if (moduleType(label) == "HLTLevel1GTSeed") {
	 const ParameterSet pset(modulePSet(label));
	 if (pset!=ParameterSet()) {
	   const bool   l1Tech(pset.getParameter<bool>("L1TechTriggerSeeding"));
	   const string l1Seed(pset.getParameter<string>("L1SeedsLogicalExpression"));
	   hltL1GTSeeds_[i].push_back(pair<bool,string>(l1Tech,l1Seed));
	 }
       }
     }
   }

   // Extract and fill streams information
   if (processPSet_.existsAs<ParameterSet>("streams",true)) {
     const ParameterSet streams(processPSet_.getParameterSet("streams"));
     streamNames_=streams.getParameterNamesForType<vector<string> >();
     sort(streamNames_.begin(),streamNames_.end());
     const unsigned int n(streamNames_.size());
     streamContents_.resize(n);
     for (unsigned int i=0; i!=n; ++i) {
       streamIndex_[streamNames_[i]]=i;
       streamContents_[i]=streams.getParameter<vector<string> >(streamNames_[i]);
       sort(streamContents_[i].begin(),streamContents_[i].end());
     }
     
   }

   // Extract and fill datasets information
   if (processPSet_.existsAs<ParameterSet>("datasets",true)) {
     const ParameterSet datasets(processPSet_.getParameterSet("datasets"));
     datasetNames_=datasets.getParameterNamesForType<vector<string> >();
     sort(datasetNames_.begin(),datasetNames_.end());
     const unsigned int n(datasetNames_.size());
     datasetContents_.resize(n);
     for (unsigned int i=0; i!=n; ++i) {
       datasetIndex_[datasetNames_[i]]=i;
       datasetContents_[i]=datasets.getParameter< vector<string> >(datasetNames_[i]);
       sort(datasetContents_[i].begin(),datasetContents_[i].end());
     }
   }

   // Extract and fill Prescale information

   // Check various possibilities to get the HLT prescale sets:
   string prescaleName("");
   const string preS("PrescaleService");
   const string preT("PrescaleTable");
   if (processPSet_.exists(preS)) {
     prescaleName=preS;
   } else if ( processPSet_.exists(preT)) {
     prescaleName=preT;
   }
   if (prescaleName=="") {
     hltPrescaleTable_=HLTPrescaleTable();
   } else {
     const ParameterSet iPS(processPSet_.getParameter<ParameterSet>(prescaleName));
     string defaultLabel(iPS.getUntrackedParameter<string>("lvl1DefaultLabel",""));
     vector<string> labels;
     if (iPS.exists("lvl1Labels")) {
       labels = iPS.getParameter<vector<string> >("lvl1Labels");
     }
     vector<ParameterSet> vpTable;
     if (iPS.exists("prescaleTable")) {
       vpTable=iPS.getParameter<vector<ParameterSet> >("prescaleTable");
     }
     unsigned int set(0);
     const unsigned int n(labels.size());
     for (unsigned int i=0; i!=n; ++i) {
       if (labels[i]==defaultLabel) set=i;
     }
     map<string,vector<unsigned int> > table;
     const unsigned int m (vpTable.size());
     for (unsigned int i=0; i!=m; ++i) {
       table[vpTable[i].getParameter<std::string>("pathName")] = 
	 vpTable[i].getParameter<std::vector<unsigned int> >("prescales");
     }
     if (n>0) {
       hltPrescaleTable_=HLTPrescaleTable(set,labels,table);
     } else {
       hltPrescaleTable_=HLTPrescaleTable();
     }

   }

   return;
}

void HLTConfigProvider::dump(const std::string& what) const {
   using namespace std;
   using namespace edm;

   if (what=="processName") {
     cout << "HLTConfigProvider::dump: ProcessName = " << processName_ << endl;
   } else if (what=="ProcessPSet") {
     cout << "HLTConfigProvider::dump: ProcessPSet = " << endl << processPSet_ << endl;
   } else if (what=="TableName") {
     cout << "HLTConfigProvider::dump: TableName = " << tableName_ << endl;
   } else if (what=="Triggers") {
     const unsigned int n(size());
     cout << "HLTConfigProvider::dump: Triggers: " << n << endl;
     for (unsigned int i=0; i!=n; ++i) {
       cout << "  " << i << " " << triggerNames_[i] << endl;
     }
   } else if (what=="TriggerSeeds") {
     const unsigned int n(size());
     cout << "HLTConfigProvider::dump: TriggerSeeds: " << n << endl;
     for (unsigned int i=0; i!=n; ++i) {
       const unsigned int m(hltL1GTSeeds_[i].size());
       cout << "  " << i << " " << triggerNames_[i] << " " << m << endl;
       for (unsigned int j=0; j!=m; ++j) {
	 cout << "    " << j
	      << " " << hltL1GTSeeds_[i][j].first
	      << "/" << hltL1GTSeeds_[i][j].second << endl;
       }
     }
   } else if (what=="Modules") {
     const unsigned int n(size());
     cout << "HLTConfigProvider::dump Triggers and Modules: " << n << endl;
     for (unsigned int i=0; i!=n; ++i) {
       const unsigned int m(size(i));
       cout << i << " " << triggerNames_[i] << " " << m << endl;
       cout << " - Modules: ";
       unsigned int nHLTPrescalers(0);
       unsigned int nHLTLevel1GTSeed(0);
       for (unsigned int j=0; j!=m; ++j) {
	 const string& label(moduleLabels_[i][j]);
	 const string  type(moduleType(label));
	 cout << " " << j << ":" << label << "/" << type ;
	 if (type=="HLTPrescaler") nHLTPrescalers++;
	 if (type=="HLTLevel1GTSeed") nHLTLevel1GTSeed++;
       }
       cout << endl;
       cout << " - Number of HLTPrescaler/HLTLevel1GTSeed modules: " 
	    << nHLTPrescalers << "/" << nHLTLevel1GTSeed << endl;
     }
   } else if (what=="StreamNames") {
     const unsigned int n(streamNames_.size());
     cout << "HLTConfigProvider::dump: StreamNames: " << n << endl;
     for (unsigned int i=0; i!=n; ++i) {
       cout << "  " << i << " " << streamNames_[i] << endl;
     }
   } else if (what=="Streams") {
     const unsigned int n(streamNames_.size());
     cout << "HLTConfigProvider::dump: Streams: " << n << endl;
     for (unsigned int i=0; i!=n; ++i) {
       const unsigned int m(streamContents_[i].size());
       cout << "  " << i << " " << streamNames_[i] << " " << m << endl;
       for (unsigned int j=0; j!=m; ++j) {
	 cout << "    " << j << " " << streamContents_[i][j] << endl;
       }
     }
   } else if (what=="DatasetNames") {
     const unsigned int n(datasetNames_.size());
     cout << "HLTConfigProvider::dump: DatasetNames: " << n << endl;
     for (unsigned int i=0; i!=n; ++i) {
       cout << "  " << i << " " << datasetNames_[i] << endl;
     }
   } else if (what=="Datasets") {
     const unsigned int n(datasetNames_.size());
     cout << "HLTConfigProvider::dump: Datasets: " << n << endl;
     for (unsigned int i=0; i!=n; ++i) {
       const unsigned int m(datasetContents_[i].size());
       cout << "  " << i << " " << datasetNames_[i] << " " << m << endl;
       for (unsigned int j=0; j!=m; ++j) {
	 cout << "    " << j << " " << datasetContents_[i][j] << endl;
       }
     }
   } else if (what=="PrescaleTable") {
     const unsigned int n (hltPrescaleTable_.size());
     cout << "HLTConfigProvider::dump: PrescaleTable: # of sets : " << n << endl;
     const vector<string>& labels(hltPrescaleTable_.labels());
     for (unsigned int i=0; i!=n; ++i) {
       cout << " " << i << "/'" << labels.at(i) << "'";
     }
     if (n>0) cout << endl;
     const map<string,vector<unsigned int> >& table(hltPrescaleTable_.table());
     cout << "HLTConfigProvider::dump: PrescaleTable: # of paths: " << table.size() << endl;
     const map<string,vector<unsigned int> >::const_iterator tb(table.begin());
     const map<string,vector<unsigned int> >::const_iterator te(table.end());
     for (map<string,vector<unsigned int> >::const_iterator ti=tb; ti!=te; ++ti) {
       for (unsigned int i=0; i!=n; ++i) {
	 cout << " " << ti->second.at(i);
       }
       cout << " " << ti->first << endl;
     }
   } else {
     cout << "HLTConfigProvider::dump: Unkown dump request: " << what << endl;
   }
   return;
}

unsigned int HLTConfigProvider::size() const {
  return triggerNames_.size();
}
unsigned int HLTConfigProvider::size(unsigned int trigger) const {
  return moduleLabels_.at(trigger).size();
}
unsigned int HLTConfigProvider::size(const std::string& trigger) const {
  return size(triggerIndex(trigger));
}

const std::string& HLTConfigProvider::tableName() const {
  return tableName_;
}
const std::vector<std::string>& HLTConfigProvider::triggerNames() const {
  return triggerNames_;
}
const std::string& HLTConfigProvider::triggerName(unsigned int trigger) const {
  return triggerNames_.at(trigger);
}
unsigned int HLTConfigProvider::triggerIndex(const std::string& trigger) const {
  const std::map<std::string,unsigned int>::const_iterator index(triggerIndex_.find(trigger));
  if (index==triggerIndex_.end()) {
    return size();
  } else {
    return index->second;
  }
}

const std::vector<std::string>& HLTConfigProvider::moduleLabels(unsigned int trigger) const {
  return moduleLabels_.at(trigger);
}
const std::vector<std::string>& HLTConfigProvider::moduleLabels(const std::string& trigger) const {
  return moduleLabels_.at(triggerIndex(trigger));
}

const std::string& HLTConfigProvider::moduleLabel(unsigned int trigger, unsigned int module) const {
  return moduleLabels_.at(trigger).at(module);
}
const std::string& HLTConfigProvider::moduleLabel(const std::string& trigger, unsigned int module) const {
  return moduleLabels_.at(triggerIndex(trigger)).at(module);
}

unsigned int HLTConfigProvider::moduleIndex(unsigned int trigger, const std::string& module) const {
  const std::map<std::string,unsigned int>::const_iterator index(moduleIndex_.at(trigger).find(module));
  if (index==moduleIndex_.at(trigger).end()) {
    return size(trigger);
  } else {
    return index->second;
  }
}
unsigned int HLTConfigProvider::moduleIndex(const std::string& trigger, const std::string& module) const {
  return moduleIndex(triggerIndex(trigger),module);
}

const std::string HLTConfigProvider::moduleType(const std::string& module) const {
  if (processPSet_.exists(module)) {
    return modulePSet(module).getParameter<std::string>("@module_type");
  } else {
    return "";
  }
}

const edm::ParameterSet& HLTConfigProvider::processPSet() const {
  return processPSet_;
}

const edm::ParameterSet HLTConfigProvider::modulePSet(const std::string& module) const {
  if (processPSet_.exists(module)) {
    return processPSet_.getParameter<edm::ParameterSet>(module);
  } else {
    return edm::ParameterSet();
  }
}

const std::vector<std::vector<std::pair<bool,std::string> > >& HLTConfigProvider::hltL1GTSeeds() const {
  return hltL1GTSeeds_;
}

const std::vector<std::pair<bool,std::string> >& HLTConfigProvider::hltL1GTSeeds(const std::string& trigger) const {
  return hltL1GTSeeds(triggerIndex(trigger));
}

const std::vector<std::pair<bool,std::string> >& HLTConfigProvider::hltL1GTSeeds(unsigned int trigger) const {
  return hltL1GTSeeds_.at(trigger);
}

/// Streams                                                                   
const std::vector<std::string>& HLTConfigProvider::streamNames() const {
  return streamNames_;
}

const std::string& HLTConfigProvider::streamName(unsigned int stream) const {
  return streamNames_.at(stream);
}


unsigned int HLTConfigProvider::streamIndex(const std::string& stream) const {
  const std::map<std::string,unsigned int>::const_iterator index(streamIndex_.find(stream));
  if (index==streamIndex_.end()) {
    return streamNames_.size();
  } else {
    return index->second;
  }
}

const std::vector<std::vector<std::string> >& HLTConfigProvider::streamContents() const {
  return streamContents_;
}

const std::vector<std::string>& HLTConfigProvider::streamContent(unsigned int stream) const {
  return streamContents_.at(stream);
}

const std::vector<std::string>& HLTConfigProvider::streamContent(const std::string& stream) const {
  return streamContent(streamIndex(stream));
}

/// Datasets                                                                  
const std::vector<std::string>& HLTConfigProvider::datasetNames() const {
  return datasetNames_;
}

const std::string& HLTConfigProvider::datasetName(unsigned int dataset) const {
  return datasetNames_.at(dataset);
}

unsigned int HLTConfigProvider::datasetIndex(const std::string& dataset) const {
  const std::map<std::string,unsigned int>::const_iterator index(datasetIndex_.find(dataset));
  if (index==datasetIndex_.end()) {
    return datasetNames_.size();
  } else {
    return index->second;
  }
}

const std::vector<std::vector<std::string> >& HLTConfigProvider::datasetContents() const {
  return datasetContents_;
}

const std::vector<std::string>& HLTConfigProvider::datasetContent(unsigned int dataset) const {
  return datasetContents_.at(dataset);
}

const std::vector<std::string>& HLTConfigProvider::datasetContent(const std::string& dataset) const {
  return datasetContent(datasetIndex(dataset));
}

unsigned int HLTConfigProvider::prescaleSize() const {
  return hltPrescaleTable_.size();
}
unsigned int HLTConfigProvider::prescaleValue(unsigned int set, const std::string& trigger) const {
  return hltPrescaleTable_.prescale(set,trigger);
}

const std::vector<std::string>& HLTConfigProvider::prescaleLabels() const {
  return hltPrescaleTable_.labels();
}
const std::map<std::string,std::vector<unsigned int> >& HLTConfigProvider::prescaleTable() const {
  return hltPrescaleTable_.table();
}

int HLTConfigProvider::prescaleSet(const edm::Event& iEvent) const {
  //  return hltPrescaleTable_.set();
  L1GtUtils l1GtUtils;
  int errTech(0),errPhys(0);
  int psfTech(0),psfPhys(0);
  psfTech=l1GtUtils.prescaleFactorSetIndex(iEvent,"TechnicalTriggers",errTech);
  psfPhys=l1GtUtils.prescaleFactorSetIndex(iEvent,"PhysicsAlgorithms",errPhys);
  if ( (errTech==0) && (errPhys==0) &&
       (psfTech>=0) && (psfPhys>=0) && (psfTech==psfPhys) ) {
    return psfPhys;
  } else {
    /// error
    return -1;
  }
}
unsigned int HLTConfigProvider::prescaleValue(const edm::Event& iEvent, const std::string& trigger) const {
  const int set(prescaleSet(iEvent));
  if (set<0) {
    return 1;
  } else {
    return prescaleValue(static_cast<unsigned int>(set),trigger);
  }
}
