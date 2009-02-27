/*----------------------------------------------------------------------

ProvenanceAdaptor.cc

----------------------------------------------------------------------*/
  //------------------------------------------------------------
  // Class ProvenanceAdaptor: adapts old provenance (fileFormatVersion_.value_ < 11) to new provenance.
  // Also adapts old parameter sets (fileFormatVersion_.value_ < 12) to new provenance.
#include <algorithm>
#include <cassert>
#include "IOPool/Input/src/ProvenanceAdaptor.h"
#include <set>
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/ParameterSet/interface/split.h"
#include "FWCore/Utilities/interface/Algorithms.h"

namespace edm {

  namespace {
    void
    insertIntoReplace(ProvenanceAdaptor::StringMap& replace,
		std::string const& fromPrefix,
		std::string const& from,
		std::string const& fromPostfix,
		std::string const& toPrefix,
		std::string const& to,
		std::string const& toPostfix) {
      replace.insert(std::make_pair(fromPrefix+from+fromPostfix, toPrefix+to+toPostfix));
    }
  }

  void
  ProvenanceAdaptor::convertParameterSets(StringWithIDList& in, StringMap& replace, ParameterSetIdConverter& psetIdConverter) {
    std::string const comma(",");
    std::string const rparam(")");
    std::string const rvparam("})");
    std::string const loldparam("=+P(");
    std::string const loldvparam("=+p({");
    std::string const lparam("=+Q(");
    std::string const lvparam("=+q({");
    bool doItAgain = false;
    for (StringMap::const_iterator j = replace.begin(), jEnd = replace.end(); j != jEnd; ++j) {
      for (StringWithIDList::iterator i = in.begin(), iEnd = in.end(); i != iEnd; ++i) {
	for (std::string::size_type it = i->first.find(j->first); it != std::string::npos; it = i->first.find(j->first)) {
	  i->first.replace(it, j->first.size(), j->second);
	  doItAgain = true;
	}
      }
    }
    for (StringWithIDList::iterator i = in.begin(), iEnd = in.end(); i != iEnd;) {
      if (i->first.find("+P") == std::string::npos && i->first.find("+p") == std::string::npos) {
        ParameterSet pset(i->first);
        pset.registerIt();
        pset.setFullyTracked();
	std::string& from = i->first;
	std::string to;
	ParameterSetID newID(pset.id());
	newID.toString(to);
	insertIntoReplace(replace, loldparam, from, rparam, lparam, to, rparam);
	insertIntoReplace(replace, comma, from, comma, comma, to, comma);
	insertIntoReplace(replace, comma, from, rvparam, comma, to, rvparam);
	insertIntoReplace(replace, loldvparam, from, comma, lvparam, to, comma);
	insertIntoReplace(replace, loldvparam, from, rvparam, lvparam, to, rvparam);
	if (i->second != newID && i->second != ParameterSetID()) {
	  psetIdConverter.insert(std::make_pair(i->second, newID));
	}
	StringWithIDList::iterator icopy = i;
	++i;
	in.erase(icopy);
	doItAgain = true;
      } else {
	++i;
      }
    }
    if (!doItAgain && !in.empty()) {
      for (StringWithIDList::iterator i = in.begin(), iEnd = in.end(); i != iEnd; ++i) {
	std::list<std::string> pieces;
	split(std::back_inserter(pieces), i->first, '<', ';', '>');
	for (std::list<std::string>::iterator i= pieces.begin(), e = pieces.end(); i != e; ++i) {
	  std::string removeName = i->substr(i->find('+'));
	  if (removeName.size() >= 4) {
	    if (removeName[1] == 'P') {
	      std::string psetString(removeName.begin()+3, removeName.end()-1);
	      in.push_back(std::make_pair(psetString, ParameterSetID()));
	      doItAgain = true;
	    } else if (removeName[1] == 'p') {
	      std::string pvec = std::string(removeName.begin()+3, removeName.end()-1);
	      StringList temp;
	      split(std::back_inserter(temp), pvec, '{', ',', '}');
	      for (StringList::const_iterator j = temp.begin(), f = temp.end(); j != f; ++j) {
		in.push_back(std::make_pair(*j, ParameterSetID()));
	      }
	      doItAgain = true;
	    }
	  }	
	}
      }
    }
    if (doItAgain) {
      convertParameterSets(in, replace, psetIdConverter);
    }
  }
  
  void
  ProvenanceAdaptor::fixProcessHistory(ProcessHistoryMap& pHistMap,
				       ProcessHistoryVector& pHistVector) {
    assert (pHistMap.empty() != pHistVector.empty());
    for (ProcessHistoryVector::const_iterator i = pHistVector.begin(), e = pHistVector.end(); i != e; ++i) {
      pHistMap.insert(std::make_pair(i->id(), *i));
    }
    pHistVector.clear();
    for (ProcessHistoryMap::const_iterator i = pHistMap.begin(), e = pHistMap.end(); i != e; ++i) {
      ProcessHistory newHist;
      ProcessHistoryID const& oldphID = i->first;
      for (ProcessHistory::const_iterator it = i->second.begin(), et = i->second.end(); it != et; ++it) {
	ParameterSetID const& newPsetID = convertID(it->parameterSetID());
	newHist.push_back(ProcessConfiguration(it->processName(), newPsetID, it->releaseVersion(), it->passID()));
      }
      assert(newHist.size() == i->second.size());
      ProcessHistoryID newphID = newHist.id();
      pHistVector.push_back(newHist);
      if (newphID != oldphID) {
        processHistoryIdConverter_.insert(std::make_pair(oldphID, newphID));
      }
    }
    assert(pHistVector.size() == pHistMap.size());
  }

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
    fillProcessConfiguration(ProcessHistoryVector const& pHistVec, ProcessConfigurationVector& procConfigVector) {
      procConfigVector.clear();
      std::set<ProcessConfiguration> pcset;
      for (ProcessHistoryVector::const_iterator it = pHistVec.begin(), itEnd = pHistVec.end();
	  it != itEnd; ++it) {
	for (ProcessConfigurationVector::const_iterator i = it->begin(), iEnd = it->end();
	    i != iEnd; ++i) {
	  if (pcset.insert(*i).second) {
	    procConfigVector.push_back(*i);
	  }
	}
      }
    }

    void
    fillMapsInProductRegistry(ProcessConfigurationVector const& procConfigVector,
			      ProductRegistry& productRegistry) {
      for (ProductRegistry::ProductList::iterator
	    it = productRegistry.productListUpdator().begin(),
	    itEnd = productRegistry.productListUpdator().end();
	    it != itEnd; ++it) {
	BranchDescription& bd = it->second;
	bd.parameterSetIDs().clear();
	bd.moduleNames().clear();
      }
      std::string const triggerResults = std::string("TriggerResults");
      std::string const source = std::string("source");
      std::string const input = std::string("@main_input");
      for (ProcessConfigurationVector::const_iterator i = procConfigVector.begin(), iEnd = procConfigVector.end();
	  i != iEnd; ++i) {
	ProcessConfigurationID pcid = i->id();
	std::string const& processName = i->processName();
	ParameterSetID const& processParameterSetID = i->parameterSetID();
	ParameterSet processParameterSet;
	pset::Registry::instance()->getMapped(processParameterSetID, processParameterSet);
        if (processParameterSet.empty()) {
          continue;
        }
	for (ProductRegistry::ProductList::iterator
	    it = productRegistry.productListUpdator().begin(),
	    itEnd = productRegistry.productListUpdator().end();
	    it != itEnd; ++it) {
	  BranchDescription& bd = it->second;
	  if (processName != bd.processName()) {
	    continue;
	  }
	  std::string const& moduleLabel = bd.moduleLabel();
	  if (moduleLabel == triggerResults) {
	    continue; // No parameter set for "TriggerResults"
	  }
	  bool isInput = (moduleLabel == source);
	  ParameterSet moduleParameterSet = processParameterSet.getParameter<ParameterSet>(isInput ? input : moduleLabel);
	  moduleParameterSet.registerIt();
	  bd.parameterSetIDs().insert(std::make_pair(pcid, moduleParameterSet.id()));
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
	     ProcessHistoryMap& pHistMap,
	     ProcessHistoryVector& pHistVector,
	     ProcessConfigurationVector& procConfigVector,
	     ParameterSetIdConverter const& parameterSetIdConverter,
	     bool fullConversion) :
	        parameterSetIdConverter_(parameterSetIdConverter),
	        processHistoryIdConverter_(),
		branchIDLists_(),
		branchListIndexes_() {
    fixProcessHistory(pHistMap, pHistVector);
    fillProcessConfiguration(pHistVector, procConfigVector);
    fillMapsInProductRegistry(procConfigVector, productRegistry);
    if (fullConversion) {
      fillListsAndIndexes(productRegistry, pHistMap, branchIDLists_, branchListIndexes_);
    }
  }

  ParameterSetID const&
  ProvenanceAdaptor::convertID(ParameterSetID const& oldID) const {
    ParameterSetIdConverter::const_iterator it = parameterSetIdConverter_.find(oldID);
    if (it == parameterSetIdConverter_.end()) {
      return oldID;
    }
    return it->second; 
  }

  ProcessHistoryID const&
  ProvenanceAdaptor::convertID(ProcessHistoryID const& oldID) const {
    ProcessHistoryIdConverter::const_iterator it = processHistoryIdConverter_.find(oldID);
    if (it == processHistoryIdConverter_.end()) {
      return oldID;
    }
    return it->second; 
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
