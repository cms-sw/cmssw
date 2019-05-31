#ifndef CondFormats_L1TObjects_L1GtTriggerMask_h
#define CondFormats_L1TObjects_L1GtTriggerMask_h

/**
 * \class L1GtTriggerMask
 * 
 * 
 * Description: L1 GT mask.  
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * 
 * $Date$
 * $Revision$
 *
 */

// system include files
#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>
#include <ostream>

// user include files
//   base class

// forward declarations

// class declaration

class L1GtTriggerMask {
public:
  // constructors
  //
  L1GtTriggerMask();

  //  from a vector
  L1GtTriggerMask(std::vector<unsigned int>&);

  // destructor
  virtual ~L1GtTriggerMask();

public:
  /// get the trigger mask
  inline const std::vector<unsigned int>& gtTriggerMask() const { return m_triggerMask; }

  /// set the trigger mask
  void setGtTriggerMask(std::vector<unsigned int>&);

  /// print the mask
  void print(std::ostream&) const;

private:
  /// trigger mask
  std::vector<unsigned int> m_triggerMask;

  COND_SERIALIZABLE;
};

#endif /*CondFormats_L1TObjects_L1GtTriggerMask_h*/
