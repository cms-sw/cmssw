#ifndef IOPool_Input_ProvenanceAdaptor_h
#define IOPool_Input_ProvenanceAdaptor_h

/*----------------------------------------------------------------------

ProvenanceAdaptor.h 

----------------------------------------------------------------------*/
#include <map>
#include <list>
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

namespace edm {

  //------------------------------------------------------------
  // Class ProvenanceAdaptor: supports file reading.

  typedef std::map<ParameterSetID, ParameterSetBlob> ParameterSetMap;
  class ProvenanceAdaptor : private boost::noncopyable {
  public:
    typedef std::list<std::string> StringList;
    typedef std::list<std::pair<std::string, ParameterSetID> > StringWithIDList;
    typedef std::map<std::string, std::string> StringMap;
    typedef std::map<ParameterSetID, ParameterSetID> ParameterSetIdConverter;
    typedef std::map<ProcessConfigurationID, ProcessConfigurationID> ProcessConfigurationIdConverter;
    typedef std::map<ProcessHistoryID, ProcessHistoryID> ProcessHistoryIdConverter;
    ProvenanceAdaptor(
	     ProductRegistry& productRegistry,
	     ProcessHistoryMap& pHistMap,
	     ProcessHistoryVector& pHistVector,
	     ProcessConfigurationVector& procConfigVector,
	     ParameterSetIdConverter const& parameterSetIdConverter,
	     bool fullConversion);
  
    boost::shared_ptr<BranchIDLists const> branchIDLists() const;

    void branchListIndexes(BranchListIndexes & indexes) const;

    static void convertParameterSets(StringWithIDList& in, StringMap& replace, ParameterSetIdConverter& psetIdConverter);

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
