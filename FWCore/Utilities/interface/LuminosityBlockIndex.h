#ifndef FWCore_Utilities_LuminosityBlockIndex_h
#define FWCore_Utilities_LuminosityBlockIndex_h
// -*- C++ -*-
//
// Package:     FWCore/Utilities
// Class  :     edm::LuminosityBlockIndex
// 
/**\class edm::LuminosityBlockIndex LuminosityBlockIndex.h "FWCore/Utilities/interface/LuminosityBlockIndex.h"

 Description: Identifies a 'slot' being used to hold an active LuminosityBlock

 Usage:
 Various APIs used this to access per LuminosityBlock information.
 It is important to realize that the same LuminosityBockIndex may be used to refer
 to different LuminosityBlocks over the lifetime of a job. A LuminosityBlockIndex
 will only get a new LuminosityBlock after the previous LuminosityBlock using
 that index has finished being used.

*/
//
// Original Author:  Chris Jones
//         Created:  Fri, 26 Apr 2013 19:39:10 GMT
// $Id$
//

// system include files

// user include files

// forward declarations
namespace edm {
  class LuminosityBlockPrincipal;
  
  class LuminosityBlockIndex
  {
    
  public:
    LuminosityBlockIndex(const LuminosityBlockIndex&) = default;
    LuminosityBlockIndex& operator=(const LuminosityBlockIndex&) = default;
    ~LuminosityBlockIndex() = default;
    
    // ---------- const member functions ---------------------
    operator unsigned int() const {
      return value_;
    }
    
    bool operator==(LuminosityBlockIndex const& iRHS) const {
      return value() == iRHS.value();
    }

    /** \return value ranging from 0 to one less than max number of simultaneous luminosityblocks.
     */
    unsigned int value() const { return value_;}
    
  private:
    ///Only the LuminosityBlockPrincipal is allowed to make one of these
    friend class LuminosityBlockPrincipal;

    explicit LuminosityBlockIndex(unsigned int iValue): value_{iValue} {}
    
    LuminosityBlockIndex()= delete;
    
    // ---------- member data --------------------------------
    unsigned int value_;
    
  };
}


#endif
