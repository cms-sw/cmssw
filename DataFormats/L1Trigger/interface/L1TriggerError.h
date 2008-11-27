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
// $Id: $
//


class L1TriggerError {
 public:
  
  /// default ctor
  L1TriggerError();
  
  /// construct from an error code
  L1TriggerError(unsigned code);

  /// dtor
  ~L1TriggerError();

  /// set error
  void setCode(int code) { code_ = code; }
  
  /// get error
  unsigned code() { return code_; }

  private:

    unsigned code_;

};

#include <vector>

typedef std::vector<L1TriggerError> L1TriggerErrorCollection;

#endif
