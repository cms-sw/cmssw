#ifndef CondIter_CondCachedIter_h
#define CondIter_CondCachedIter_h


#include "CondCore/Utilities/interface/CondIter.h"


template <class T>
class CondCachedIter : public CondIter<T> {
public:
  CondCachedIter() : CondIter<T>(true){}
  virtual ~CondCachedIter(){}
  
};

#endif
