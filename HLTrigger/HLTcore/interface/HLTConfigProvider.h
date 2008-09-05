#ifndef HLTcore_HLTConfigProvider_h
#define HLTcore_HLTConfigProvider_h

/** \class HLTConfigProvider
 *
 *  
 *  This class provides access routines to get hold of the HLT Configuration
 *
 *  $Date: 2008/07/08 07:06:23 $
 *  $Revision: 1.9 $
 *
 *  \author Martin Grunewald
 *
 */

#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include<string>
#include<vector>

//
// class declaration
//

class HLTConfigProvider {
  
 public:

  bool init(const std::string& processName);

  void dump(const std::string& what) const;

  unsigned int size() const;
  unsigned int size(unsigned int trigger) const;
  unsigned int size(const std::string& trigger) const;

  const std::vector<std::string>& triggerNames() const;
  const std::string& triggerName(unsigned int triggerIndex) const;
  unsigned int triggerIndex(const std::string& triggerName) const;

  const std::string& moduleLabel(const std::string& trigger, unsigned int module) const;
  const std::string& moduleLabel(unsigned int trigger, unsigned int module) const;
  const std::vector<std::string>& moduleLabels(const std::string& trigger) const;
  const std::vector<std::string>& moduleLabels(unsigned int trigger) const;
  unsigned int moduleIndex(const std::string& trigger, const std::string& module) const;
  unsigned int moduleIndex(unsigned int trigger, const std::string& module) const;

  const std::string moduleType(const std::string& module) const;
  const edm::ParameterSet modulePSet(const std::string& module) const;

 private:
  std::string processName_;
  const edm::pset::Registry * registry_;

  edm::ParameterSet ProcessPSet_;

  std::vector<std::string> triggerNames_;
  std::vector<std::vector<std::string> > moduleLabels_;

  std::map<std::string,unsigned int> triggerIndex_;
  std::vector<std::map<std::string,unsigned int> > moduleIndex_;

  std::vector<std::string> pathNames_;
  std::vector<std::string> endpathNames_;

};
#endif
