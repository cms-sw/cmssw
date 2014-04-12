#ifndef MCMatchSelector_h
#define MCMatchSelector_h
/* \class MCMatchSelector
 *
 * Extended version of MCTruthPairSelector. Preselects matches
 * based on charge, pdgId and status.
 */

#include <set>
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace reco {
  template<typename T1, typename T2>
  class MCMatchSelector {
  public:
    MCMatchSelector(const edm::ParameterSet& cfg) : 
      checkCharge_(cfg.getParameter<bool>("checkCharge")) { 
      std::vector<int> ids = 
	cfg.getParameter< std::vector<int> >("mcPdgId");
      for ( std::vector<int>::const_iterator i=ids.begin();
	    i!=ids.end(); ++i )  ids_.insert(*i);
      std::vector<int> status = 
	cfg.getParameter< std::vector<int> >("mcStatus");
      for ( std::vector<int>::const_iterator i=status.begin();
	    i!=status.end(); ++i )  status_.insert(*i);
    }
    /// true if match is possible
    bool operator()( const T1 & c, const T2 & mc ) const {
      if ( checkCharge_ && c.charge() != mc.charge() ) return false;
      if ( !ids_.empty() ) {
	if ( ids_.find(abs(mc.pdgId()))==ids_.end() )  return false;
      }
      if ( status_.empty() )  return true;
      return status_.find(mc.status())!=status_.end();
    }
  private:
    bool checkCharge_;
    std::set<int> ids_;
    std::set<int> status_;
  };
}

#endif
