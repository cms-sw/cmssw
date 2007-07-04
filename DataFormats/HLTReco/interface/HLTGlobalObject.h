#ifndef HLTReco_HLTGlobalObject_h
#define HLTReco_HLTGlobalObject_h

/** \class reco::HLTGlobalObject
 *
 *  A single object in each event carrying persistent references to
 *  all HLTPathObjects available for this event (ie, usually the few
 *  triggers which accepted the event).
 *
 *  If the user wants map-like indexing of triggers through their
 *  names as key, s/he must use the TriggerNamesService.
 *
 *  $Date: 2007/07/03 07:39:22 $
 *  $Revision: 1.4 $
 *
 *  \author Martin Grunewald
 *
 */

#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/HLTReco/interface/HLTPathObject.h"
#include <vector>

namespace reco
{
  class HLTGlobalObject {

  private:
    /// vector of Refs to path objects
    std::vector<edm::RefProd<HLTPathObject> > refs_;

  public:

    /// trivial constructor
    HLTGlobalObject(): refs_() {}

    /// constructor with capacity argument
    HLTGlobalObject(unsigned int n): refs_() {refs_.reserve(n);}

    /// number of currently stored path objects
    unsigned int size() const { return refs_.size();}

    /// add new path object
    void put (const edm::RefProd<HLTPathObject>& ref) {
      refs_.push_back(ref);
    }

    /// get Ref to ith path object
    const edm::RefProd<HLTPathObject>& at (const unsigned int index) const {
      return refs_.at(index);
    }
    
  };

}

#endif
