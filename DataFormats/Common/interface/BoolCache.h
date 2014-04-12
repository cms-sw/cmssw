#ifndef DataFormats_Common_BoolCache_h
#define DataFormats_Common_BoolCache_h
// -*- C++ -*-
//
// Package:     Common
// Class  :     BoolCache
// 
/**\class BoolCache BoolCache.h DataFormats/Common/interface/BoolCache.h

 Description: ROOT safe cache flag

 Usage:
    We define schema evolution rules for this class in order to guarantee that isCached_
    is always reset to false when ever a new instance of this class is read from a file

*/
//
// Original Author:  Chris Jones
//         Created:  Sat Aug 18 17:30:08 EDT 2007
//

// system include files

// user include files

// forward declarations
namespace edm {
class BoolCache {
public:
  BoolCache() : isCached_(false) {}
  BoolCache(bool iValue) : isCached_(iValue) {}
  operator bool() { return isCached_; }
  BoolCache & operator=( bool b ) { isCached_ = b; return *this; }
private:
  bool isCached_;
};

}
#endif
