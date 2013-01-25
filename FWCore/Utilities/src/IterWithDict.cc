#include "FWCore/Utilities/interface/IterWithDict.h"
#include <cassert>

namespace edm {

  IterWithDictBase::IterWithDictBase() : iter_(static_cast<TList*>(nullptr)), atEnd_(true) {}

  IterWithDictBase::IterWithDictBase(TList* list) : iter_(list), atEnd_(list->GetSize() == 0) {
   //With a TIter, you must call Next() once to point to the first element.
    if(!atEnd_) iter_.Next();
  }

  bool
  IterWithDictBase::operator!=(IterWithDictBase const & other) const {
    // This just compares the iterator with the end of the range.
    assert(other.atEnd_);
    return !atEnd_;
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
