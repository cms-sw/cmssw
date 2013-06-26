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
 * $Date$
 * $Revision$
 *
 */

// this class header
#include "CondFormats/L1TObjects/interface/L1GtPrescaleFactors.h"

// system include files

// user include files
//   base class

// forward declarations

// constructor
L1GtPrescaleFactors::L1GtPrescaleFactors()
{
    // empty
}

L1GtPrescaleFactors::L1GtPrescaleFactors(const std::vector<std::vector<int> >& factorValue)
{
    m_prescaleFactors = factorValue;
}

// destructor
L1GtPrescaleFactors::~L1GtPrescaleFactors()
{
    // empty
}

// set the prescale factors
void L1GtPrescaleFactors::setGtPrescaleFactors(const std::vector<std::vector<int> >& factorValue)
{

    m_prescaleFactors = factorValue;

}

// print the prescale factors
void L1GtPrescaleFactors::print(std::ostream& myOstream) const {
    myOstream << "\nL1 GT Trigger prescale factors" << std::endl;

    for (unsigned iSet = 0; iSet < m_prescaleFactors.size(); iSet++) {

        myOstream << "\n\n Set index " << iSet << "\n " << std::endl;
        for (unsigned i = 0; i < (m_prescaleFactors[iSet]).size(); i++) {

            myOstream 
                << "  Bit number \t" << i 
                << ":\t prescale factor: " << (m_prescaleFactors[iSet])[i] 
                << std::endl;
        }
    }

}
