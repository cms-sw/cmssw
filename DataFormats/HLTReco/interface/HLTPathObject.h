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
 *  $Date: 2007/07/03 07:39:22 $
 *  $Revision: 1.4 $
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

    /// index (slot number) of path in trigger table - from zero
    unsigned int path_;
    /// vector of Refs to filter objects placed by HLT filters:
    std::vector<edm::RefToBase<HLTFilterObjectBase> > refs_;

  public:

    /// trivial constructor
    HLTPathObject(): path_(), refs_() { }

    /// constructor
    HLTPathObject(unsigned int p): path_(p), refs_() { }

    /// how many Refs to filter objects are stored
    unsigned int size() const { return refs_.size();}

    /// path index of path whose filter objects are stored
    unsigned int path() const { return path_;}

    /// add another filter object
    void put (const edm::RefToBase<HLTFilterObjectBase>& ref) {
      refs_.push_back(ref);
    }

    /// get the Ref to the ith filter object
    const edm::RefToBase<HLTFilterObjectBase>& at (const unsigned int index) const {
      return refs_.at(index);
    }
    
  };

}

#endif
