/**
 * \class L1GtPrescaleFactors
 * 
 * 
 * Description: L1 GT prescale factors.  
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
#include "CondFormats/L1TObjects/interface/L1GtPrescaleFactors.h"

// system include files
#include <vector>
#include <ostream>

// user include files
//   base class

// forward declarations

// constructor
L1GtPrescaleFactors::L1GtPrescaleFactors()
{
    // empty
}

L1GtPrescaleFactors::L1GtPrescaleFactors(std::vector<int>& factorValue)
{
    m_prescaleFactors = factorValue;
}

// destructor
L1GtPrescaleFactors::~L1GtPrescaleFactors()
{
    // empty
}

// set the prescale factors
void L1GtPrescaleFactors::setGtPrescaleFactors(std::vector<int>& factorValue)
{

    m_prescaleFactors = factorValue;

}

// print the prescale factors
void L1GtPrescaleFactors::print(std::ostream& s) const
{
    s << "\nL1 GT Trigger prescale factors" << std::endl;

    for (unsigned i = 0; i < m_prescaleFactors.size(); i++) {

        s << "  Bit number \t" << i
        << ":\t prescale factor: " << m_prescaleFactors[i] << std::endl;

    }

}
