#include "FWCore/Utilities/interface/IterWithDict.h"
#include <cassert>

namespace edm {

  IterWithDictBase::IterWithDictBase() : iter_(static_cast<TList*>(nullptr)), atEnd_(true) {
    // This ctor is used by the framework for the end of a range,
    // or for any type that does not have a TClass.
    // An iterator constructed by this ctor must not be used
    // as the left hand argument of operator!=().
  }

  IterWithDictBase::IterWithDictBase(TList* list) : iter_(list), atEnd_(false) {
    // With a TIter, you must call Next() once to point to the first element.
    advance();
  }

  bool
  IterWithDictBase::operator!=(IterWithDictBase const& other) const {
    // The special cases are needed because TIter::operator!=()
    // dereferences a null pointer (i.e. segfaults) if the left hand TIter
    // was constucted with a nullptr argument (the first constructor above).
    if(atEnd_ != other.atEnd_) {
      // one iterator at end, but not both
      return true; 
    } else if(atEnd_) {
      // both iterators at end
      return false;
    }
    // neither iterator at end
    return iter() != other.iter();
  }

  void
  IterWithDictBase::advance() {
    if(!atEnd_) {
      TObject* obj = iter_.Next();
      if(obj == nullptr) atEnd_ = true;
    }
  }

  TIter const&
  IterWithDictBase::iter() const {
    return iter_;
  }

}
