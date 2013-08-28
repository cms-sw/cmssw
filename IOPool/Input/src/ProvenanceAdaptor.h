#ifndef IOPool_Input_ProvenanceAdaptor_h
#define IOPool_Input_ProvenanceAdaptor_h

/*----------------------------------------------------------------------

ProvenanceAdaptor.h 

----------------------------------------------------------------------*/
#include <map>
#include <memory>
#include <vector>
#include "DataFormats/Provenance/interface/BranchIDList.h"
#include "DataFormats/Provenance/interface/BranchListIndex.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/ProvenanceFwd.h"
#include "FWCore/ParameterSet/interface/ParameterSetConverter.h"

namespace edm {

  //------------------------------------------------------------
  // Class ProvenanceAdaptor: 

  class ProvenanceAdaptor {
  public:
    typedef ParameterSetConverter::ParameterSetIdConverter ParameterSetIdConverter;
    typedef std::map<ProcessHistoryID, ProcessHistoryID> ProcessHistoryIdConverter;
    ProvenanceAdaptor(
	     ProductRegistry& productRegistry,
	     ProcessHistoryMap& pHistMap,
	     ProcessHistoryVector& pHistVector,
	     ProcessConfigurationVector& procConfigVector,
	     ParameterSetIdConverter const& parameterSetIdConverter,
	     bool fullConversion);
    ~ProvenanceAdaptor();
  
    ProvenanceAdaptor(ProvenanceAdaptor const&) = delete; // Disallow copying and moving
    ProvenanceAdaptor& operator=(ProvenanceAdaptor const&) = delete; // Disallow copying and moving

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
