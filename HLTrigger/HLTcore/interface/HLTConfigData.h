#ifndef HLTcore_HLTConfigData_h
#define HLTcore_HLTConfigData_h

/** \class HLTConfigData
 *
 *  
 *  This class provides access routines to get hold of the HLT Configuration
 *
 *
 *  \author Martin Grunewald
 *
 */

#include "DataFormats/HLTReco/interface/HLTPrescaleTable.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include<map>
#include<string>
#include<vector>

//
// class declaration
//

class HLTConfigData {

 public:
  HLTConfigData();
  HLTConfigData(const edm::ParameterSet* iID);

 private:
  /// extract information into data members - called by init() methods
  void extract();

 public:

  /// Dumping config info to cout
  void dump(const std::string& what) const;

  /// Accessors (const methods)

  /// process name
  const std::string& processName() const;

  /// GlobalTag.globaltag
  const std::string& globalTag() const;

  /// HLT ConfDB table name
  const std::string& tableName() const;

  /// number of trigger paths in trigger table
  unsigned int size() const;
  /// number of modules on a specific trigger path
  unsigned int size(unsigned int trigger) const;
  unsigned int size(const std::string& trigger) const;

  /// names of trigger paths
  const std::vector<std::string>& triggerNames() const;
  const std::string& triggerName(unsigned int triggerIndex) const;

  /// slot position of trigger path in trigger table (0 - size-1)
  unsigned int triggerIndex(const std::string& triggerName) const;

  /// label(s) of module(s) on a trigger path
  const std::vector<std::string>& moduleLabels(unsigned int trigger) const;
  const std::vector<std::string>& moduleLabels(const std::string& trigger) const;
  const std::vector<std::string>& saveTagsModules(unsigned int trigger) const;
  const std::vector<std::string>& saveTagsModules(const std::string& trigger) const;
  const std::string& moduleLabel(unsigned int trigger, unsigned int module) const;
  const std::string& moduleLabel(const std::string& trigger, unsigned int module) const;

  /// slot position of module on trigger path (0 - size-1)
  unsigned int moduleIndex(unsigned int trigger, const std::string& module) const;
  unsigned int moduleIndex(const std::string& trigger, const std::string& module) const;

  /// C++ class name of module
  const std::string moduleType(const std::string& module) const;

  /// C++ base class name of module
  const std::string moduleEDMType(const std::string& module) const;

  /// ParameterSet of process
  const edm::ParameterSet& processPSet() const;

  /// ParameterSet of module
  const edm::ParameterSet& modulePSet(const std::string& module) const;

  /// Is module an L3 filter (ie, tracked saveTags=true)
  bool saveTags(const std::string& module) const;


  /// L1T type (0=unknown, 1=legacy/stage-1 or 2=stage-2)                                                                                    
  unsigned int l1tType() const;

  /// HLTLevel1GTSeed module
  /// HLTLevel1GTSeed modules for all trigger paths
  const std::vector<std::vector<std::pair<bool,std::string> > >& hltL1GTSeeds() const;
  /// HLTLevel1GTSeed modules for trigger path with name
  const std::vector<std::pair<bool,std::string> >& hltL1GTSeeds(const std::string& trigger) const;
  /// HLTLevel1GTSeed modules for trigger path with index i
  const std::vector<std::pair<bool,std::string> >& hltL1GTSeeds(unsigned int trigger) const;

  /// HLTL1TSeed module
  /// HLTL1TSeed modules for all trigger paths
  const std::vector<std::vector<std::string> >& hltL1TSeeds() const;
  /// HLTL1TSeed modules for trigger path with name
  const std::vector<std::string>& hltL1TSeeds(const std::string& trigger) const;
  /// HLTL1TSeed modules for trigger path with index i
  const std::vector<std::string>& hltL1TSeeds(unsigned int trigger) const;


  /// Streams
  /// list of names of all streams
  const std::vector<std::string>& streamNames() const;
  /// name of stream with index i
  const std::string& streamName(unsigned int stream) const;
  /// index of stream with name
  unsigned int streamIndex(const std::string& stream) const;
  /// names of datasets for all streams
  const std::vector<std::vector<std::string> >& streamContents() const;
  /// names of datasets in stream with index i
  const std::vector<std::string>& streamContent(unsigned int stream) const;
  /// names of datasets in stream with name
  const std::vector<std::string>& streamContent(const std::string& stream) const;


  /// Datasets
  /// list of names of all datasets
  const std::vector<std::string>& datasetNames() const;
  /// name of dataset with index i
  const std::string& datasetName(unsigned int dataset) const;
  /// index of dataset with name
  unsigned int datasetIndex(const std::string& dataset) const;
  /// names of trigger paths for all datasets
  const std::vector<std::vector<std::string> >& datasetContents() const;
  /// names of trigger paths in dataset with index i
  const std::vector<std::string>& datasetContent(unsigned int dataset) const;
  /// names of trigger paths in dataset with name
  const std::vector<std::string>& datasetContent(const std::string& dataset) const;


  /// HLT prescale info
  /// Number of HLT prescale sets
  unsigned int prescaleSize() const;
  /// HLT prescale value in specific prescale set for a specific trigger path
  unsigned int prescaleValue(unsigned int set, const std::string& trigger) const;
  /// low-level data member access 
  const std::vector<std::string>& prescaleLabels() const;
  const std::map<std::string,std::vector<unsigned int> >& prescaleTable() const;

  /// technical: id() function needed for use with ThreadSafeRegistry
  edm::ParameterSetID id() const;

 private:

  const edm::ParameterSet* processPSet_;

  std::string processName_;
  std::string globalTag_;
  std::string tableName_;
  std::vector<std::string> triggerNames_;
  std::vector<std::vector<std::string> > moduleLabels_;
  std::vector<std::vector<std::string> > saveTagsModules_;

  std::map<std::string,unsigned int> triggerIndex_;
  std::vector<std::map<std::string,unsigned int> > moduleIndex_;

  unsigned int l1tType_;
  std::vector<std::vector<std::pair<bool,std::string> > > hltL1GTSeeds_;
  std::vector<std::vector<std::string> > hltL1TSeeds_;

  std::vector<std::string> streamNames_;
  std::map<std::string,unsigned int> streamIndex_;
  std::vector<std::vector<std::string> > streamContents_;

  std::vector<std::string> datasetNames_;
  std::map<std::string,unsigned int> datasetIndex_;
  std::vector<std::vector<std::string> > datasetContents_;

  trigger::HLTPrescaleTable hltPrescaleTable_;

};
#endif
