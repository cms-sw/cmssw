#ifndef IOPool_Input_ProvenanceAdaptor_h
#define IOPool_Input_ProvenanceAdaptor_h

/*----------------------------------------------------------------------

ProvenanceAdaptor.h 

----------------------------------------------------------------------*/
#include <map>
#include <list>
#include <set>
#include <memory>
#include <string>
#include <vector>
#include "boost/shared_ptr.hpp"
#include "boost/utility.hpp"
#include "DataFormats/Provenance/interface/ProvenanceFwd.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Provenance/interface/ProcessConfigurationRegistry.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/ParameterSetBlob.h"
#include "DataFormats/Provenance/interface/ProvenanceFwd.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {
  typedef std::vector<std::string> StringVector;

  struct MainParameterSet {
    MainParameterSet(ParameterSetID const& oldID, std::string const& psetString);
    ~MainParameterSet();
    ParameterSetID oldID_;
    ParameterSet parameterSet_;
    StringVector paths_;
    StringVector endPaths_;
    std::set<std::string> triggerPaths_;
  };

  struct TriggerPath {
    TriggerPath(ParameterSet const& pset);
    ~TriggerPath();
    ParameterSet parameterSet_;
    StringVector tPaths_;
    std::set<std::string> triggerPaths_;
  };

  //------------------------------------------------------------
  // Class ParameterSetConverter

  typedef std::map<ParameterSetID, ParameterSetBlob> ParameterSetMap;
  class ParameterSetConverter : private boost::noncopyable {
  public:
    typedef std::list<std::string> StringList;
    typedef std::map<std::string, std::string> StringMap;
    typedef std::list<std::pair<std::string, ParameterSetID> > StringWithIDList;
    typedef std::map<ParameterSetID, ParameterSetID> ParameterSetIdConverter;
    ParameterSetConverter(ParameterSetMap const& psetMap, ParameterSetIdConverter& idConverter, bool alreadyByReference);
    ~ParameterSetConverter();
    ParameterSetIdConverter const& parameterSetIdConverter() const {return parameterSetIdConverter_;}
  private:
    void convertParameterSets();
    void noConvertParameterSets();
    StringWithIDList parameterSets_;
    std::vector<MainParameterSet> mainParameterSets_;
    std::vector<TriggerPath> triggerPaths_;
    StringMap replace_;
    ParameterSetIdConverter& parameterSetIdConverter_;
  };

  //------------------------------------------------------------
  // Class ProvenanceAdaptor: 

  class ProvenanceAdaptor : private boost::noncopyable {
  public:
    typedef ParameterSetConverter::ParameterSetIdConverter ParameterSetIdConverter;
    typedef std::map<ProcessConfigurationID, ProcessConfigurationID> ProcessConfigurationIdConverter;
    typedef std::map<ProcessHistoryID, ProcessHistoryID> ProcessHistoryIdConverter;
    ProvenanceAdaptor(
	     ProductRegistry& productRegistry,
	     ProcessHistoryMap& pHistMap,
	     ProcessHistoryVector& pHistVector,
	     ProcessConfigurationVector& procConfigVector,
	     ParameterSetIdConverter const& parameterSetIdConverter,
	     bool fullConversion);
    ~ProvenanceAdaptor();
  
    boost::shared_ptr<BranchIDLists const> branchIDLists() const;

    void branchListIndexes(BranchListIndexes & indexes) const;

    ParameterSetID const&
    convertID(ParameterSetID const& oldID) const;

    ProcessHistoryID const&
    convertID(ProcessHistoryID const& oldID) const;

  private:
    void fixProcessHistory(ProcessHistoryMap& pHistMap,
			   ProcessHistoryVector& pHistVector);

    ParameterSetIdConverter parameterSetIdConverter_;
    ProcessHistoryIdConverter processHistoryIdConverter_;
    boost::shared_ptr<BranchIDLists const> branchIDLists_;
    std::vector<BranchListIndex> branchListIndexes_;
  }; // class ProvenanceAdaptor


}
#endif
