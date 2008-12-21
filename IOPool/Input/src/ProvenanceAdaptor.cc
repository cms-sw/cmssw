/*----------------------------------------------------------------------

ProvenanceAdaptor.cc

----------------------------------------------------------------------*/
#include <algorithm>
#include <cassert>
#include "IOPool/Input/src/ProvenanceAdaptor.h"
#include <set>
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/Utilities/interface/Algorithms.h"

namespace edm {

  
  //------------------------------------------------------------
  // Class ProvenanceAdaptor: adapts old provenance (fileFormatVersion_.value_ < 11) to new provenance.
  namespace {
    typedef std::vector<std::string> OneHistory;
    typedef std::set<OneHistory> Histories;
    typedef std::pair<std::string, BranchID> Product;
    typedef std::vector<Product> OrderedProducts;
    struct Sorter {
      explicit Sorter(Histories const& histories) : histories_(histories) {}
      bool operator()(Product const& a, Product const& b) const;
      Histories const histories_;
    };
    bool Sorter::operator()(Product const& a, Product const& b) const {
      assert (a != b);
      if (a.first == b.first) return false;
      bool mayBeTrue = false;
      for (Histories::const_iterator it = histories_.begin(), itEnd = histories_.end(); it != itEnd; ++it) {
	OneHistory::const_iterator itA = find_in_all(*it, a.first);
	if (itA == it->end()) continue;
	OneHistory::const_iterator itB = find_in_all(*it, b.first);
	if (itB == it->end()) continue;
	assert (itA != itB);
	if (itB < itA) {
	  // process b precedes process a;
	  return false;
	}
	// process a precedes process b;
	mayBeTrue = true;
      }
      return mayBeTrue;
    }

    void
    fillProcessConfigurationMap(ProcessHistoryMap const& pHistMap, ProcessConfigurationMap& procConfigMap) {
      for (ProcessHistoryMap::const_iterator it = pHistMap.begin(), itEnd = pHistMap.end();
	  it != itEnd; ++it) {
        for (std::vector<ProcessConfiguration>::const_iterator i = it->second.begin(), iEnd = it->second.end();
	    i != iEnd; ++i) {
	  procConfigMap.insert(std::make_pair(i->id(), *i));
	}
      }
    }

    void
    fillMapsInProductRegistry(ProcessConfigurationMap const& procConfigMap, ProductRegistry& productRegistry) {
      for (ProcessConfigurationMap::const_iterator i = procConfigMap.begin(), iEnd = procConfigMap.end();
	  i != iEnd; ++i) {
	ProcessConfigurationID const& pcid = i->first;
	std::string const& processName = i->second.processName();
	ParameterSetID const& processParameterSetID = i->second.parameterSetID();
	ParameterSet processParameterSet;
	pset::Registry::instance()->getMapped(processParameterSetID, processParameterSet);
	for (ProductRegistry::ProductList::iterator
	    it = productRegistry.productListUpdator().begin(),
	    itEnd = productRegistry.productListUpdator().end();
	    it != itEnd; ++it) {
	  BranchDescription& bd = it->second;
	  if (processName != bd.processName()) {
	    continue;
	  }
	  std::string const& moduleLabel = bd.moduleLabel();
	  if (moduleLabel == std::string("TriggerResults")) {
	    continue; // No parameter set for "TriggerResults"
	  }
	  ParameterSet moduleParameterSet = processParameterSet.getParameter<ParameterSet>(moduleLabel);
	  bd.parameterSetIDs().insert(std::make_pair(pcid, moduleParameterSet.trackedID()));
	  bd.moduleNames().insert(std::make_pair(pcid, moduleParameterSet.getParameter<std::string>("@module_type")));
	}
      }
    }

    void
    fillListsAndIndexes(ProductRegistry const& productRegistry,
			ProcessHistoryMap const& pHistMap,
			boost::shared_ptr<BranchIDLists const>& branchIDLists,
			std::vector<BranchListIndex>& branchListIndexes) {
      OrderedProducts orderedProducts;
      std::set<std::string> processNamesThatProduced;
      ProductRegistry::ProductList const& prodList = productRegistry.productList();
      for (ProductRegistry::ProductList::const_iterator it = prodList.begin(), itEnd = prodList.end();
	  it != itEnd; ++it) {
        if (it->second.branchType() == InEvent) {
	  it->second.init();
	  processNamesThatProduced.insert(it->second.processName());
	  orderedProducts.push_back(std::make_pair(it->second.processName(), it->second.branchID()));
        }
      }
      assert (!orderedProducts.empty());
      Histories processHistories;
      size_t max = 0;
      for(ProcessHistoryMap::const_iterator it = pHistMap.begin(), itEnd = pHistMap.end(); it != itEnd; ++it) {
        ProcessHistory const& pHist = it->second;
        OneHistory processHistory;
        for(ProcessHistory::const_iterator i = pHist.begin(), iEnd = pHist.end(); i != iEnd; ++i) {
	  if (processNamesThatProduced.find(i->processName()) != processNamesThatProduced.end()) {
	    processHistory.push_back(i->processName());
	  }
        }
        max = (processHistory.size() > max ? processHistory.size() : max);
        assert(max <= processNamesThatProduced.size());
        if (processHistory.size() > 1) {
          processHistories.insert(processHistory);
        }
      }
      stable_sort_all(orderedProducts, Sorter(processHistories));

      std::auto_ptr<BranchIDLists> pv(new BranchIDLists);
      std::auto_ptr<BranchIDList> p(new BranchIDList);
      std::string processName;
      BranchListIndex blix = 0;
      for (OrderedProducts::const_iterator it = orderedProducts.begin(), itEnd = orderedProducts.end(); it != itEnd; ++it) {
        bool newvector = it->first != processName && !processName.empty();
        if (newvector) {
	  pv->push_back(*p);
	  branchListIndexes.push_back(blix);
	  ++blix;
	  processName = it->first;
	  p.reset(new BranchIDList);
        }
        p->push_back(it->second.id());
      }
      pv->push_back(*p);
      branchListIndexes.push_back(blix);
      branchIDLists.reset(pv.release());
    }
  }

  ProvenanceAdaptor::ProvenanceAdaptor(
	     ProductRegistry& productRegistry,
	     ProcessHistoryMap const& pHistMap,
	     ProcessConfigurationMap& procConfigMap) :
		productRegistry_(productRegistry),
		branchIDLists_(),
		branchListIndexes_() {
    fillProcessConfigurationMap(pHistMap, procConfigMap);
    fillMapsInProductRegistry(procConfigMap, productRegistry);
    fillListsAndIndexes(productRegistry, pHistMap, branchIDLists_, branchListIndexes_);
  }

  boost::shared_ptr<BranchIDLists const>
  ProvenanceAdaptor::branchIDLists() const {
    return branchIDLists_;
  }

  void
  ProvenanceAdaptor::branchListIndexes(BranchListIndexes & indexes)  const {
    indexes = branchListIndexes_;
  }
}
