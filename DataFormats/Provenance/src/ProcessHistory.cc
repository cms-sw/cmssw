#include <iterator>
#include <ostream>
#include <sstream>
#include "FWCore/Utilities/interface/Digest.h"
#include "FWCore/Utilities/interface/Algorithms.h"

#include "DataFormats/Provenance/interface/ProcessHistory.h"


namespace edm {
  namespace {
    // Is rh a subset of lh.
    bool isSubsetOf(ProcessHistory const& lh, ProcessHistory const& rh) {
      if (rh.empty()) {
	return true;
      }
      if (lh.size() < rh.size()) {
	return false;
      }
      if (lh.size() == rh.size()) {
	return (lh == rh);
      }
      ProcessHistory::const_iterator j = lh.begin(), jEnd = lh.end();
      for (ProcessHistory::const_iterator i = rh.begin(), iEnd = rh.end(); i != iEnd; ++i) {
	while(j != jEnd && j->processName() != i->processName()) {
	  ++j;
        }
	if (j == jEnd) {
	  return false;
	}
	if (*i != *j) {
	  return false;
	}
      }
      return true;
    }
  }
  ProcessHistoryID
  ProcessHistory::id() const {
    if(phid().isValid()) {
      return phid();
    }
    // This implementation is ripe for optimization.
    // We do not use operator<< because it does not write out everything.
    std::ostringstream oss;
    for (const_iterator i = begin(), e = end(); i != e; ++i) {
      oss << i->processName() << ' '
	  << i->parameterSetID() << ' ' 
	  << i->releaseVersion() << ' '
	  << i->passID() << ' ';
    }
    std::string stringrep = oss.str();
    cms::Digest md5alg(stringrep);
    ProcessHistoryID tmp(md5alg.digest().toString());
    phid().swap(tmp);
    return phid();
  }

  bool
  ProcessHistory::getConfigurationForProcess(std::string const& name, 
					     ProcessConfiguration& config) const {
    for (const_iterator i = begin(), e = end(); i != e; ++i) {
      if (i->processName() == name) {
	config = *i;
	return true;
      }
    }
    // Name not found!
    return false;				    
  }

  // Used only for backward compatibility
  bool
  ProcessHistory::mergeProcessHistory(ProcessHistory const& other) {
    if (isSubsetOf(*this, other)) {
      return true;
    } else if (isSubsetOf(other, *this)) {
      *this = other;
      return true;
    }
    return false;
  }

  bool
  isAncestor(ProcessHistory const& a, ProcessHistory const& b) {
    if (a.size() >= b.size()) return false;
    typedef ProcessHistory::collection_type::const_iterator const_iterator;
    for (const_iterator itA = a.data().begin(), itB = b.data().begin(),
         itAEnd = a.data().end(); itA != itAEnd; ++itA, ++itB) {
      if (*itA != *itB) return false;
    }
    return true;
  }

  std::ostream&
  operator<<(std::ostream& ost, ProcessHistory const& ph) {
    ost << "Process History = ";
    copy_all(ph, std::ostream_iterator<ProcessHistory::value_type>(ost,";"));
    return ost;
  }
}
