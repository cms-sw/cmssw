#ifndef IOPool_Input_ProvenanceAdaptor_h
#define IOPool_Input_ProvenanceAdaptor_h

/*----------------------------------------------------------------------

ProvenanceAdaptor.h 

----------------------------------------------------------------------*/
#include <map>
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
  ProvenanceAdaptor(
	     ProductRegistry& productRegistry,
	     ProcessHistoryMap const& pHistMap,
	     ProcessConfigurationVector& procConfigVector);

  
  boost::shared_ptr<BranchIDLists const> branchIDLists() const;

  void branchListIndexes(BranchListIndexes & indexes) const;

  private:
    ProductRegistry const& productRegistry_;
    boost::shared_ptr<BranchIDLists const> branchIDLists_;
    std::vector<BranchListIndex> branchListIndexes_;
  }; // class ProvenanceAdaptor

}
#endif
