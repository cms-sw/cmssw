#ifndef FWCore_Utilities_HideStdSharedPtrFromRoot_h
#define FWCore_Utilities_HideStdSharedPtrFromRoot_h
#if defined(__GCCXML__)
// -*- C++ -*-
//
// Package:     FWCore/Utilities
// Class  :     HideStdSharedPtrFromRoot
// 
/**

 Description: gccxml can't parse std::shared_ptr so we need to give a substitute

 Usage:
    gccxml has its own version of <memory> which does not include 
 std::shared_ptr<> so we need a declaration of the class which has
 the same memory footprint and declares those methods which are seen
 by gccxml.

*/
//
// Original Author:  Chris Jones
//         Created:  Wed, 04 Dec 2013 16:39:12 GMT
//

// system include files

// user include files

// forward declarations
namespace std {
  template< typename T>
  class shared_ptr {
    void* data_;
    unsigned long count_;
  public:
    shared_ptr();
    shared_ptr(T*);
    void reset(T* iValue=0);
    T* get();
    T const* get() const;
    
    T* operator->();
    T const* operator->() const;

    T operator*() const;
    
    operator bool() const;
  };
}
#endif
#endif
