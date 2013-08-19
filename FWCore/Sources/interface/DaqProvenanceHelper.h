#ifndef FWCore_Sources_DaqProvenanceHelper_h
#define FWCore_Sources_DaqProvenanceHelper_h

#include <map>
#include <string>
#include <vector>

#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/ParentageID.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "DataFormats/Provenance/interface/ProductProvenance.h"
#include "DataFormats/Provenance/interface/BranchID.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {
  class BranchChildren;
  struct DaqProvenanceHelper {
    typedef std::map<ProcessHistoryID, ProcessHistoryID> ProcessHistoryIDMap;
    typedef std::map<ParentageID, ParentageID> ParentageIDMap;
    explicit DaqProvenanceHelper(TypeID const& rawDataType);
    ProcessHistoryID daqInit(ProductRegistry& productRegistry) const;
    void saveInfo(BranchDescription const& oldBD, BranchDescription const& newBD) {
      oldProcessName_ = oldBD.processName();
      oldBranchID_ = oldBD.branchID();
      newBranchID_ = newBD.branchID();
    }
    bool matchProcesses(ProcessConfiguration const& pc, ProcessHistory const& ph) const;
    void fixMetaData(std::vector<ProcessConfiguration>& pcv, std::vector<ProcessHistory>& phv);
    void fixMetaData(std::vector<BranchID>& branchIDs) const;
    void fixMetaData(BranchIDLists const&) const;
    void fixMetaData(BranchChildren& branchChildren) const;
    ProcessHistoryID const& mapProcessHistoryID(ProcessHistoryID const& phid);
    ParentageID const& mapParentageID(ParentageID const& phid) const;
    BranchID const& mapBranchID(BranchID const& branchID) const;

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
