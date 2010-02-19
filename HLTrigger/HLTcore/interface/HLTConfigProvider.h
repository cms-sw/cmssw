#ifndef HLTcore_HLTConfigProvider_h
#define HLTcore_HLTConfigProvider_h

/** \class HLTConfigProvider
 *
 *  
 *  This class provides access routines to get hold of the HLT Configuration
 *
 *  $Date: 2010/02/18 15:07:16 $
 *  $Revision: 1.18 $
 *
 *  \author Martin Grunewald
 *
 */

#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include<string>
#include<vector>

//
// class declaration
//

class HLTConfigProvider {
  
 public:

  /// init methods - use one and only one!

  /// very old, deprecated, may fail when processing file(s) containing
  /// events accepted by different HLT tables!
  bool init(const std::string& processName);

  /// old, deprecated as well
  /// the parameter "changed" indicates whether the config has
  /// actually changed
  bool init(const edm::Event& iEvent,                                              const std::string& processName, bool& changed);

  /// new, revised, based on advice by Chris Jones (Feb.2010)
  /// call from beginRun
  bool init(const edm::Run& iRun,                   const edm::EventSetup& iSetup, const std::string& processName, bool& changed);


 private:

  /// call from beginRun
  bool init(const edm::Run& iRun,                                                  const std::string& processName, bool& changed);

  /// call from beginLuminsoityBlock
  bool init(const edm::LuminosityBlock& iLumiBlock,                                const std::string& processName, bool& changed);
  bool init(const edm::LuminosityBlock& iLumiBlock, const edm::EventSetup& iSetup, const std::string& processName, bool& changed);

  /// call from produce/filter/analyze method
  bool init(const edm::Event& iEvent,               const edm::EventSetup& iSetup, const std::string& processName, bool& changed);

  /// real init method 
  bool init(const edm::ProcessHistory& iHistory, const std::string& processName, bool& changed);
  bool init(const edm::ProcessHistory& iHistory, const edm::EventSetup& iSetup, const std::string& processName, bool& changed);
  /// clear data members - called by init() method
  void clear();
  /// extract information into data members - called by init() method
  void extract();


 public:
  /// dump config aspects to cout
  void dump(const std::string& what) const;

  /// accessors

  /// number of trigger paths in trigger table
  unsigned int size() const;
  /// number of modules on a specific trigger path
  unsigned int size(unsigned int trigger) const;
  unsigned int size(const std::string& trigger) const;

  /// HLT ConfDB table name
  const std::string& tableName() const;

  /// names of trigger paths
  const std::vector<std::string>& triggerNames() const;
  const std::string& triggerName(unsigned int triggerIndex) const;
  /// slot position of trigger path in trigger table (0 - size-1)
  unsigned int triggerIndex(const std::string& triggerName) const;

  /// label(s) of module(s) on a trigger path
  const std::vector<std::string>& moduleLabels(unsigned int trigger) const;
  const std::vector<std::string>& moduleLabels(const std::string& trigger) const;
  const std::string& moduleLabel(unsigned int trigger, unsigned int module) const;
  const std::string& moduleLabel(const std::string& trigger, unsigned int module) const;

  /// slot position of module on trigger path (0 - size-1)
  unsigned int moduleIndex(unsigned int trigger, const std::string& module) const;
  unsigned int moduleIndex(const std::string& trigger, const std::string& module) const;

  /// C++ class name of module
  const std::string moduleType(const std::string& module) const;

  /// ParameterSet of process
  const edm::ParameterSet& processPSet() const;

  /// ParameterSet of module
  const edm::ParameterSet modulePSet(const std::string& module) const;


  /// HLTLevel1GTSeed module
  /// HLTLevel1GTSeed modules for all trigger paths
  const std::vector<std::vector<std::pair<bool,std::string> > >& hltL1GTSeeds() const;
  /// HLTLevel1GTSeed modules for trigger path with name
  const std::vector<std::pair<bool,std::string> >& hltL1GTSeeds(const std::string& trigger) const;
  /// HLTLevel1GTSeed modules for trigger path with index i
  const std::vector<std::pair<bool,std::string> >& hltL1GTSeeds(unsigned int trigger) const;


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


  /*  Not useable: PrescaleService configuration is not saved in Provenance
  /// PrescaleService accessors

  /// Available prescale column labels
  const std::vector<std::string>& prescaleLabels() const;

  /// Prescale column label of given index key
  const std::string& prescaleLabel(unsigned int label) const;

  /// Index key of given column label
  unsigned int prescaleIndex(const std::string& label) const;

  /// Prescale values for given trigger
  const std::vector<unsigned int>& prescaleValues(unsigned int trigger) const;

  /// Prescale values for given trigger
  const std::vector<unsigned int>& prescaleValues(const std::string& trigger) const;

  /// Prescale value for given trigger and prescale index key
  unsigned int prescaleValue(unsigned int trigger, unsigned int label) const;

  /// Prescale value for given trigger and prescale label
  unsigned int prescaleValue(const std::string& trigger, const std::string& label) const;
  */

  /// c'tor
  HLTConfigProvider():
    processName_(""), processPSet_(),
    tableName_(), triggerNames_(), moduleLabels_(),
    triggerIndex_(), moduleIndex_(),
    hltL1GTSeeds_(),
    streamNames_(), streamIndex_(), streamContents_(),
    datasetNames_(), datasetIndex_(), datasetContents_(),
    prescaleLabels_(), prescaleIndex_(), prescaleValues_() { }

 private:

  std::string processName_;
  edm::ParameterSet processPSet_;

  std::string tableName_;
  std::vector<std::string> triggerNames_;
  std::vector<std::vector<std::string> > moduleLabels_;

  std::map<std::string,unsigned int> triggerIndex_;
  std::vector<std::map<std::string,unsigned int> > moduleIndex_;

  std::vector<std::vector<std::pair<bool,std::string> > > hltL1GTSeeds_;

  std::vector<std::string> streamNames_;
  std::map<std::string,unsigned int> streamIndex_;
  std::vector<std::vector<std::string> > streamContents_;

  std::vector<std::string> datasetNames_;
  std::map<std::string,unsigned int> datasetIndex_;
  std::vector<std::vector<std::string> > datasetContents_;

  std::vector<std::string> prescaleLabels_;
  std::map<std::string,unsigned int> prescaleIndex_;
  std::vector<std::vector<unsigned int> > prescaleValues_;

};
#endif
