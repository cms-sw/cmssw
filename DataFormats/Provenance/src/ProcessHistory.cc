#include <iterator>
#include <ostream>
#include <sstream>
#include "FWCore/Utilities/interface/Digest.h"
#include "FWCore/Utilities/interface/Algorithms.h"

#include "DataFormats/Provenance/interface/ProcessHistory.h"

namespace edm {
  ProcessHistoryID
  ProcessHistory::id() const {
    if(transient_.phid_.isValid()) {
      return transient_.phid_;
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
    ProcessHistoryID phID(md5alg.digest().toString());
    return phID;
  }

  ProcessHistoryID
  ProcessHistory::setProcessHistoryID() {
    if(!transient_.phid_.isValid()) {
      transient_.phid_ = id();
    }
    return transient_.phid_;
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

  void
  ProcessHistory::reduce() {
    phid() = ProcessHistoryID();
    for (iterator i = data_.begin(), e = data_.end(); i != e; ++i) {
      i->reduce();
    }
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
