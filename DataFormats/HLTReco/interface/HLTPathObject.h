#ifndef HLTReco_HLTPathObject_h
#define HLTReco_HLTPathObject_h

/** \class HLTPathObject
 *
 *  
 *  This class contains the HLT path object, ie, combining the HLT
 *  filter objects of the filters on this path.  At the moment this is
 *  a very simple but generic solution, exploiting the fact that all
 *  HLT filter objects are of the same C++ type (HLTFilterObject).
 *  This also allows the same simple generic path collector module to
 *  work for all possible paths
 *
 *  $Date: 2006/04/11 10:10:10 $
 *  $Revision: 1.0 $
 *
 *  \author Martin Grunewald
 *
 */

#include "DataFormats/HLTReco/interface/HLTFilterObject.h"
#include <map>

namespace reco
{
  template <typename T>
  class HLTPathObject {

  private:
    // map of: filter position (=index on path), filter object
    map <unsigned char, T>  filterobjects_;

  public:
    HLTPathObject(): filterobjects_() { }

    // accessors

    unsigned int size() const { return filterobjects_.size();}

    void put (const unsigned int index, const T& filterobject) {
      filterobjects_[index]=filterobject;
    }

    const T get (const unsigned int index) const {
      if (filterobjects_.find(index)!=filterobjects_.end()) {
        return filterobjects_.find(index)->second;
      } else {
	return T();
      }
    }
    
  };

}

#endif
