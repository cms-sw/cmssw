#ifndef FWCore_Utilities_IterWithDict_h
#define FWCore_Utilities_IterWithDict_h

/*----------------------------------------------------------------------

IterWithDict:  An iterator for a TList so a range for loop can be used

----------------------------------------------------------------------*/

#include "TList.h"

namespace edm {

class IterWithDictBase {
private:
  TIter iter_;
  bool atEnd_;
protected:
  void advance();
  TIter const& iter() const;
public:
  IterWithDictBase();
  explicit IterWithDictBase(TList*);
  bool operator!=(IterWithDictBase const&) const;
};

template<typename T>
class IterWithDict : public IterWithDictBase {
public:
  IterWithDict() : IterWithDictBase() {}
  explicit IterWithDict(TList* list) : IterWithDictBase(list) {}
  IterWithDict<T>& operator++() {
    advance();
    return *this;
  }
  T* operator*() const {
    return static_cast<T*>(*iter());
  }
};

} // namespace edm

#endif // FWCore_Utilities_IterWithDict_h
