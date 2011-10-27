#ifndef DataFormats_Common_ConstPtrCache_h
#define DataFormats_Common_ConstPtrCache_h
// -*- C++ -*-
//
// Package:     Common
// Class  :     PtrCache
// 
/**\class ConstPtrCache ConstPtrCache.h DataFormats/Common/interface/ConstPtrCache.h

 Description: ROOT safe cache of a pointer

 Usage:
We define schema evolution rules for this class in order to guarantee that ptr_
is always reset to 0 when ever a new instance of this class is read from a file

*/
//
// Original Author:  Chris Jones
//         Created:  Sat Aug 18 17:30:04 EDT 2007
//

// system include files

// user include files

// forward declarations
namespace edm {
class ConstPtrCache
{

   public:
   ConstPtrCache() : ptr_(0) {}
  ConstPtrCache(const void* iPtr) : ptr_(iPtr) {}
  
   const void* ptr_;
};

}
#endif
