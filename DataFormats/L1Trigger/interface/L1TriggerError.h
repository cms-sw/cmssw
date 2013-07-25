#ifndef L1TriggerError_h
#define L1TriggerError_h

// -*- C++ -*-
//
// Package:     DataFormatsL1Trigger
// Class  :     L1TriggerError
// 
/**\class L1TriggerError \file L1TriggerError.h DataFormats/L1Trigger/interface/L1TriggerError.h \author Jim Brooke

 Description: Class for communicating errors between modules.
              Intended to be transient *only*
*/
//
// Original Author:  Jim Brooke
//         Created:  
// $Id: L1TriggerError.h,v 1.3 2009/09/18 15:08:26 jbrooke Exp $
//


class L1TriggerError {
 public:
  
  /// construct from an error code
  explicit L1TriggerError(unsigned short prodID=0, unsigned short code=0);

  /// dtor
  ~L1TriggerError();

  /// set error
  void setCode(int code) { code_ = code; }
  
  /// get error
  unsigned code() { return code_; }

  /// producer ID
  unsigned prodID();

  /// producer error
  unsigned prodErr();

  private:

    unsigned code_;

};

#include <vector>

typedef std::vector<L1TriggerError> L1TriggerErrorCollection;

#endif
