#ifndef CMSSW_mayown_ptr_H
#define CMSSW_mayown_ptr_H

#include<cassert>
#include<cstring>

// a smart pointer which may own 
// can be implemented trivially with shared_ptr
// this is infinetely ligher
// assume alignment > 2....

#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
template<typename T, int N=sizeof(T*)>
class mayown_ptr {
private:
  T const * p=nullptr;

  void markOwn() {
    if (nullptr==p) return;
    unsigned char buff[N]; memcpy(buff,&p,N);
    assert((buff[N-1]&1)==0);
    ++buff[N-1];
    memcpy(&p,buff,N);
  }
public:
  bool isOwn() const {
    unsigned char buff[N]; memcpy(buff,&p,N);
    return 1==(buff[N-1]&1);
  }
private:
  T const * pointer() const {
    unsigned char buff[N]; memcpy(buff,&p,N);
    buff[N-1] &= 0xFE;
    assert((buff[N-1]&1)==0);
    T const * np;
    memcpy(&np,buff,N);
    return np;
  }

  void destroy() {
    if (isOwn()) delete const_cast<T*>(pointer());
  }
public:
  explicit mayown_ptr(T * ip=nullptr) : p(ip) { markOwn();}
  explicit mayown_ptr(T const & ip) : p(&ip) {}
  ~mayown_ptr() { destroy();}
  mayown_ptr(mayown_ptr &)=delete;
  mayown_ptr(mayown_ptr && rh) : p(rh.p) { rh.p=nullptr;} 
  mayown_ptr& operator=(mayown_ptr &)=delete;
  mayown_ptr& operator=(mayown_ptr && rh) { destroy(); p=rh.p; rh.p=nullptr; return *this;}

  T const & operator*() const { return *pointer();}
  T const * operator->() const { return pointer();} 	
  T const * get() const { return pointer();}
  T const * release() { auto np=pointer(); p=nullptr; return np;}
  void reset() { destroy(); p=nullptr;}              
  void reset(T * ip) { destroy(); p=ip; markOwn();}
  void reset(T const & ip) { destroy(); p=&ip;}
  bool empty() const { return nullptr==p;}

  T const * raw() const { return p;}
};

template<typename T>
bool operator==(mayown_ptr<T> const & rh, mayown_ptr<T> const & lh) {
  return rh.raw() == lh.raw();
}  
template<typename T>
bool operator<(mayown_ptr<T> const & rh, mayown_ptr<T> const & lh) {
  return rh.raw() < lh.raw();
}  
#else
template<typename T, int N=sizeof(T*)>
class mayown_ptr {
  mayown_ptr() : p(nullptr) {}
  ~mayown_ptr() {}
private:
  T const * p;
  mayown_ptr(mayown_ptr &){}
  mayown_ptr& operator=(mayown_ptr &){}
};
#endif

#endif
