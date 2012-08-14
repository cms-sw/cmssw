#ifndef CondIter_CondIter_h
#define CondIter_CondIter_h


#include "CondCore/Utilities/interface/CondBasicIter.h"

#include "CondCore/DBCommon/interface/PayloadRef.h"

#include <vector>


template <class DataT>
class CondIter : public  CondBasicIter{
  
protected:
  virtual bool load(cond::DbSession& sess, std::string const & itoken) {
    if (useCache)
      if (n>=cache.size()) {
	cache.resize(n+1); 
	return cache.back().load(sess,itoken);
      } else return true;
    else return data.load(sess,itoken);
  }
  
private:
  bool initialized;
  bool useCache;
  cond::PayloadRef<DataT> data;
  std::vector<cond::PayloadRef<DataT> > cache;
  size_t n;

public:
  
  
  CondIter(bool cacheIt=false) : initialized(false), useCache(cacheIt),n(0){}
  virtual ~CondIter(){}
  
  
  void reset() { initialized=false; data.clear();}

  void rewind() { reset();}

  virtual void clear() {
    reset();
    cache.clear();
  }
  
 
  /**
     Obtain the pointer to an object T. If it is the last T the method returns a null pointer.
  */ 
  DataT const * next() {
    bool ok=false;
    if (!initialized) {
      n=0;
      ok =init();
      initialized=true;
    }
    else {
      ++n;
      ok = forward();
    }
    if (!ok) return 0;
    ok = make();
    if (!ok) return 0;
    return  useCache ?  &(*cache[n]) : &(*data); 

  }

};




#endif

