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
 * $Date:$
 * $Revision:$
 *
 */

// this class header
#include "CondFormats/L1TObjects/interface/L1GtTriggerMask.h"

// system include files
#include <vector>
#include <ostream>

// user include files
//   base class

// constructors
L1GtTriggerMask::L1GtTriggerMask()
{
    //empty - all value set by default to zero
}

L1GtTriggerMask::L1GtTriggerMask(std::vector<unsigned int>& maskValue)
{
    m_triggerMask = maskValue;
}

// destructor
L1GtTriggerMask::~L1GtTriggerMask()
{
    // empty
}

// set the trigger mask
void L1GtTriggerMask::setGtTriggerMask(std::vector<unsigned int>& maskValue)
{

    m_triggerMask = maskValue;

}

// print the mask
void L1GtTriggerMask::print(std::ostream& s) const
{
    s << "\nL1 GT Trigger mask" << std::endl;

    for (unsigned i = 0; i < m_triggerMask.size(); i++) {
        s << "  Bit number " << i << ":\t mask: " << m_triggerMask[i] << std::endl;
    }

}
