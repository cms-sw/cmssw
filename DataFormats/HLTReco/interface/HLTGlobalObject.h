#ifndef HLTReco_HLTGlobalObject_h
#define HLTReco_HLTGlobalObject_h

/** \class HLTGlobalObject
 *
 *  A single object in each event carrying persistent references to
 *  all HLTPathObjects available for this event (ie, usually the few
 *  triggers which accepted the event).
 *
 *  If the user wants map-like indexing of triggers through their
 *  names as key, s/he must use the TriggerNamesService.
 *
 *  $Date: 2006/04/26 09:27:44 $
 *  $Revision: 1.1 $
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
    std::vector<edm::RefProd<HLTPathObject> > refs_;

  public:

    // constructors

    HLTGlobalObject(): refs_() {}

    // accessors

    unsigned int size() const { return refs_.size();}

    void put (const edm::RefProd<HLTPathObject>& ref) {
      refs_.push_back(ref);
    }

    const edm::RefProd<HLTPathObject>& at (const unsigned int index) const {
      return refs_.at(index);
    }
    
  };

}

#endif
