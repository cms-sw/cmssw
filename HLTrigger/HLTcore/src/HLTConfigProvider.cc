/** \class HLTConfigProvider
 *
 * See header file for documentation
 *
 *  $Date: 2008/09/19 07:26:03 $
 *  $Revision: 1.3 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include <cassert>
#include <iostream>

bool HLTConfigProvider::init(const std::string& processName)
{
   using namespace std;
   using namespace edm;

   // clear and initialise

   processName_ = processName;
   registry_    = pset::Registry::instance();

   ProcessPSet_ = ParameterSet();
   tableName_   = "/dev/null";

   triggerNames_.clear();
   moduleLabels_.clear();

   triggerIndex_.clear();
   moduleIndex_.clear();

   pathNames_.clear();
   endpathNames_.clear();

   // Obtain ParameterSetID for requested process (with name
   // processName) from pset registry
   ParameterSetID ProcessPSetID;
   for (edm::pset::Registry::const_iterator i = registry_->begin(); i != registry_->end(); ++i) {
     if (i->second.exists("@process_name") and i->second.getParameter<string>("@process_name") == processName_)
       ProcessPSetID = i->first;
   }
   if (ProcessPSetID==ParameterSetID()) return false;

   // Obtain ParameterSet from ParameterSetID
   if (!(registry_->getMapped(ProcessPSetID,ProcessPSet_))) return false;

   // Obtain HLT PSet containing table name (available only in 2_1_10++ files)
   if (ProcessPSet_.exists("HLTConfigVersion")) {
     const ParameterSet HLTPSet(ProcessPSet_.getParameter<ParameterSet>("HLTConfigVersion"));
     if (HLTPSet.exists("tableName")) {
       tableName_=HLTPSet.getParameter<string>("tableName");
     }
   }
   //cout << "HLTConfigProvider::init() HLT-ConfDB TableName = " << tableName_ << endl;

   // Extract trigger paths, which are paths but with endpaths to be
   // removed, from ParameterSet
   pathNames_   = ProcessPSet_.getParameter<vector<string> >("@paths");
   endpathNames_= ProcessPSet_.getParameter<vector<string> >("@end_paths");
   const unsigned int nP(pathNames_.size());
   const unsigned int nE(endpathNames_.size());
   for (unsigned int iE=0; iE!=nE; ++iE) {
     const std::string& endpath(endpathNames_[iE]);
     for (unsigned int iP=0; iP!=nP; ++iP) {
       if (pathNames_[iP]==endpath) {
	 pathNames_[iP]=""; // erase endpaths
       }
     }
   }
   triggerNames_.reserve(nP);
   for (unsigned int iP=0; iP!=nP; ++iP) {
     if (pathNames_[iP]!="") {triggerNames_.push_back(pathNames_[iP]);}
   }
   pathNames_   = ProcessPSet_.getParameter<vector<string> >("@paths");

   // Obtain module labels of all modules on all trigger paths
   const unsigned int n(size());
   moduleLabels_.reserve(n);
   for (unsigned int i=0;i!=n; ++i) {
     moduleLabels_.push_back(ProcessPSet_.getParameter<vector<string> >(triggerNames_[i]));
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

   return true;
}

void HLTConfigProvider::dump (const std::string& what) const {
   using namespace std;
   using namespace edm;

   if (what=="processName") {
     cout << "HLTConfigProvider::dump: ProcessName = " << processName_ << endl;
   } else if (what=="ProcessPSet") {
     cout << "HLTConfigProvider::dump: ProcessPSet = " << endl << ProcessPSet_ << endl;
   } else if (what=="TableName") {
     cout << "HLTConfigProvider::dump: TableName = " << tableName_ << endl;
   } else if (what=="Triggers") {
     cout << "HLTConfigProvider::dump: Triggers: " << endl;
     const unsigned int n(size());
     for (unsigned int i=0; i!=n; ++i) {
       cout << "  " << i << " " << triggerNames_[i] << endl;
     }
   } else if (what=="Modules") {
     cout << "HLTConfigProvider::dump Triggers and Modules: " << endl;
     const unsigned int n(size());
     for (unsigned int i=0; i!=n; ++i) {
       cout << i << " " << triggerNames_[i] << endl;
       const unsigned int m(size(i));
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
  if (ProcessPSet_.exists(module)) {
    return modulePSet(module).getParameter<std::string>("@module_type");
  } else {
    return "";
  }
}

const edm::ParameterSet HLTConfigProvider::modulePSet(const std::string& module) const {
  if (ProcessPSet_.exists(module)) {
    return ProcessPSet_.getParameter<edm::ParameterSet>(module);
  } else {
    return edm::ParameterSet();
  }
}
