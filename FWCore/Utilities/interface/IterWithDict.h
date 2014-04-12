#ifndef FWCore_Utilities_IterWithDict_h
#define FWCore_Utilities_IterWithDict_h

/*----------------------------------------------------------------------
  
IterWithDict:  An iterator for a TList so a range for loop can be used

----------------------------------------------------------------------*/

#include "TList.h"

namespace edm {

  class IterWithDictBase {
  public:
    IterWithDictBase();
    explicit IterWithDictBase(TList* list);
    bool operator!=(IterWithDictBase const &) const;

  protected:
    void advance();
    TIter const& iter() const;

  private:
    TIter iter_;
    bool atEnd_;
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
}
#endif
