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

// this class header
#include "CondFormats/L1TObjects/interface/L1GtTriggerMask.h"

// system include files
#include <iostream>
#include <iomanip>

// user include files
//   base class

// constructors
L1GtTriggerMask::L1GtTriggerMask() {
  //empty - all value set by default to zero
}

L1GtTriggerMask::L1GtTriggerMask(std::vector<unsigned int>& maskValue) { m_triggerMask = maskValue; }

// destructor
L1GtTriggerMask::~L1GtTriggerMask() {
  // empty
}

// set the trigger mask
void L1GtTriggerMask::setGtTriggerMask(std::vector<unsigned int>& maskValue) { m_triggerMask = maskValue; }

// print the mask
void L1GtTriggerMask::print(std::ostream& outputStream) const {
  outputStream << "\nL1 GT Trigger masks are printed for all L1 partitions. "
               << "\n  Partition numbering: partition \"i\" -> bit i"
               << " (bit 0 is LSB)\n"
               << "\n If mask value is 1 for a given algorithm/technical trigger in a given partition "
               << "\n then the algorithm/technical trigger is masked (has value 0 = false) in the evaluation "
               << "\n of FinalOR.\n"
               << "\n For veto masks, if the mask is set to 1 and the result of the trigger for that bit is true, "
               << "\n then the FinalOR is set to false (no L1A).\n"
               << std::endl;

  for (unsigned i = 0; i < m_triggerMask.size(); i++) {
    outputStream << "  Algorithm/technical trigger bit number " << std::setw(3) << i << ":\t mask: 0x" << std::hex
                 << std::setw(2) << m_triggerMask[i] << std::dec << std::endl;
  }
}
