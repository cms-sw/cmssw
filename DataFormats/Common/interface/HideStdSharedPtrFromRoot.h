#ifndef DataFormats_Common_HideStdSharedPtrFromRoot_h
#define DataFormats_Common_HideStdSharedPtrFromRoot_h
#if defined(__CINT__) || defined(__MAKECINT__) || defined(__REFLEX__)
// -*- C++ -*-
//
// Package:     DataFormats/Common
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
    void reset(T* iValue=0);
    T* get();
    T const* get() const;
    
    T* operator->();
    T const* operator->() const;
    
    operator bool() const;
  };
}
#endif
#endif
