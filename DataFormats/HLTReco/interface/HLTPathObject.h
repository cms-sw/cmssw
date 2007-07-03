#ifndef HLTReco_HLTPathObject_h
#define HLTReco_HLTPathObject_h

/** \class reco::HLTPathObject
 *
 *  
 *  This class contains the HLT path object, ie, combining the HLT
 *  filter objects of the filters on this path.  At the moment this is
 *  a very simple but generic solution, exploiting the fact that all
 *  HLT filter objects are derived from the same base class
 *  HLTFilterObjectBase.
 *
 *  $Date: 2006/06/17 20:17:01 $
 *  $Revision: 1.3 $
 *
 *  \author Martin Grunewald
 *
 */

#include "DataFormats/HLTReco/interface/HLTFilterObject.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include<vector>

namespace reco
{
  class HLTPathObject {

  private:

    unsigned int path_; // index of path on trigger table
    // vector of Refs to filter objects:
    std::vector<edm::RefToBase<HLTFilterObjectBase> > refs_;

  public:
    HLTPathObject(): path_(), refs_() { }
    HLTPathObject(unsigned int p): path_(p), refs_() { }

    unsigned int size() const { return refs_.size();}

    // accessors

    unsigned int path() const { return path_;}

    void put (const edm::RefToBase<HLTFilterObjectBase>& ref) {
      refs_.push_back(ref);
    }

    const edm::RefToBase<HLTFilterObjectBase>& at (const unsigned int index) const {
      return refs_.at(index);
    }
    
  };

}

#endif
