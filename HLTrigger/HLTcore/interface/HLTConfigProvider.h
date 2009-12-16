#ifndef HLTcore_HLTConfigProvider_h
#define HLTcore_HLTConfigProvider_h

/** \class HLTConfigProvider
 *
 *  
 *  This class provides access routines to get hold of the HLT Configuration
 *
 *  $Date: 2009/12/15 16:25:24 $
 *  $Revision: 1.8 $
 *
 *  \author Martin Grunewald
 *
 */

#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include<string>
#include<vector>

//
// class declaration
//

class HLTConfigProvider {
  
 public:

  /// init for every Event!
  bool init(const std::string& processName); // <<== deprecated!
  bool init(const edm::Event& iEvent, const std::string& processName);
  /// clear data members - called by init() methods, not by user!
  void clear();
  /// extract information into data members - called by init() methods
  void extract();

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
    processName_(""), registry_(), processPSet_(),
    tableName_(), triggerNames_(), moduleLabels_(),
    triggerIndex_(), moduleIndex_(),
    pathNames_(), endpathNames_(),
    prescaleLabels_(), prescaleIndex_(), prescaleValues_() { }

 private:

  std::string processName_;

  const edm::pset::Registry * registry_;

  edm::ParameterSet processPSet_;

  std::string tableName_;
  std::vector<std::string> triggerNames_;
  std::vector<std::vector<std::string> > moduleLabels_;

  std::map<std::string,unsigned int> triggerIndex_;
  std::vector<std::map<std::string,unsigned int> > moduleIndex_;

  std::vector<std::string> pathNames_;
  std::vector<std::string> endpathNames_;

  std::vector<std::string> prescaleLabels_;
  std::map<std::string,unsigned int> prescaleIndex_;
  std::vector<std::vector<unsigned int> > prescaleValues_;

};
#endif
