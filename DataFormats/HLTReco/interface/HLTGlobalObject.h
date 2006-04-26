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
 *  $Date: 2006/04/11 10:10:10 $
 *  $Revision: 1.0 $
 *
 *  \author Martin Grunewald
 *
 */

#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/HLTReco/interface/HLTPathObject.h"
#include <map>

namespace reco
{
  template <typename T>  // T is HLTPathObject type
  class HLTGlobalObject {

  // Ref to EDProduct containing HLTPathObject
  typedef edm::RefProd<T> HLTPathObjectRef;

  // Path (trigger) index in trigger table,  ref to HLTPathObject
  typedef     map<unsigned int,  HLTPathObjectRef>  HLTPathObjectRefMap;

  private:
    HLTPathObjectRefMap pathobjectrefs_;

  public:

    // constructors

    HLTGlobalObject(): pathobjectrefs_() {}

    // accessors

    unsigned int size() const { return pathobjectrefs_.size();}

    void put (const unsigned int index, const HLTPathObjectRef& ref) {
      pathobjectrefs_[index]=ref;
    }

    const HLTPathObjectRef get (const unsigned int index) const {
      if (pathobjectrefs_.find(index)!=pathobjectrefs_.end()) {
        return pathobjectrefs_.find(index)->second;
      } else {
	return HLTPathObjectRef();
      }
    }
    
  };

}

#endif
