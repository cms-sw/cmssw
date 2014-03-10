#ifndef FWCore_Sources_DaqProvenanceHelper_h
#define FWCore_Sources_DaqProvenanceHelper_h

#include <map>
#include <string>
#include <vector>
#include "tbb/concurrent_unordered_map.h"


#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/ParentageID.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "DataFormats/Provenance/interface/ProductProvenance.h"
#include "DataFormats/Provenance/interface/BranchID.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {
  class BranchChildren;
  class ProcessHistoryRegistry;

  namespace dqh {
    struct parentage_hash {
      std::size_t operator()(edm::ParentageID const& iKey) const{
        return iKey.smallHash();
      }
    };
  }

  struct DaqProvenanceHelper {
    typedef std::map<ProcessHistoryID, ProcessHistoryID> ProcessHistoryIDMap;
    typedef tbb::concurrent_unordered_map<ParentageID, ParentageID, dqh::parentage_hash> ParentageIDMap;
    explicit DaqProvenanceHelper(TypeID const& rawDataType);
    ProcessHistoryID daqInit(ProductRegistry& productRegistry, ProcessHistoryRegistry& processHistoryRegistry) const;
    void saveInfo(BranchDescription const& oldBD, BranchDescription const& newBD) {
      oldProcessName_ = oldBD.processName();
      oldBranchID_ = oldBD.branchID();
      newBranchID_ = newBD.branchID();
    }
    bool matchProcesses(ProcessConfiguration const& pc, ProcessHistory const& ph) const;
    void fixMetaData(ProcessConfigurationVector& pcv, std::vector<ProcessHistory>& phv);
    void fixMetaData(std::vector<BranchID>& branchIDs) const;
    void fixMetaData(BranchIDLists const&) const;
    void fixMetaData(BranchChildren& branchChildren) const;
    ProcessHistoryID const& mapProcessHistoryID(ProcessHistoryID const& phid);
    ParentageID const& mapParentageID(ParentageID const& phid) const;
    BranchID const& mapBranchID(BranchID const& branchID) const;

    BranchDescription const& branchDescription() const {return constBranchDescription_;}
    ProcessHistoryID const* oldProcessHistoryID() const { return oldProcessHistoryID_; }
    ProductProvenance const& dummyProvenance() const { return dummyProvenance_; }

    void setOldParentageIDToNew(ParentageID const& iOld, ParentageID const& iNew);

  private:
    BranchDescription const constBranchDescription_;
    ProductProvenance dummyProvenance_;
    ParameterSet processParameterSet_;

    std::string oldProcessName_;
    BranchID oldBranchID_;
    BranchID newBranchID_;
    ProcessHistoryID const* oldProcessHistoryID_;
    ProcessHistoryIDMap phidMap_;
    ParentageIDMap parentageIDMap_;
  };
}
#endif
