#include "IOPool/Provenance/interface/CommonProvenanceFiller.h"

#include "DataFormats/Provenance/interface/BranchType.h"
#include "DataFormats/Provenance/interface/ParameterSetBlob.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"

#include "TBranch.h"
#include "TTree.h"

#include <cassert>
#include <utility>

namespace edm {

  void fillParameterSetBranch(TTree* parameterSetsTree, int basketSize) {
    std::pair<ParameterSetID, ParameterSetBlob> idToBlob;
    std::pair<ParameterSetID, ParameterSetBlob>* pIdToBlob = &idToBlob;
    TBranch* b =
        parameterSetsTree->Branch(poolNames::idToParameterSetBlobsBranchName().c_str(), &pIdToBlob, basketSize, 0);

    for (auto const& pset : *pset::Registry::instance()) {
      idToBlob.first = pset.first;
      idToBlob.second.pset() = pset.second.toString();

      b->Fill();
    }
  }

  void fillProcessHistoryBranch(TTree* metaDataTree,
                                int basketSize,
                                ProcessHistoryRegistry const& processHistoryRegistry) {
    ProcessHistoryVector procHistoryVector;
    for (auto const& ph : processHistoryRegistry) {
      procHistoryVector.push_back(ph.second);
    }
    ProcessHistoryVector* p = &procHistoryVector;
    TBranch* b = metaDataTree->Branch(poolNames::processHistoryBranchName().c_str(), &p, basketSize, 0);
    assert(b);
    b->Fill();
  }

}  // namespace edm
