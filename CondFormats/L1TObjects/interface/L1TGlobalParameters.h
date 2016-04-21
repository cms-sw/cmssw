#ifndef CondFormats_L1TObjects_L1TGlobalParameters_h
#define CondFormats_L1TObjects_L1TGlobalParameters_h

/**
 * \class L1TGlobalParameters
 * 
 * 
 * Description: L1 GT stable parameters.  
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
#include <vector>

#include <ostream>

#include <boost/cstdint.hpp>

// user include files
//   base class

// forward declarations

// class declaration
class L1TGlobalParameters
{

public:

    // constructor
    L1TGlobalParameters();

    // destructor
    ~L1TGlobalParameters();

public:

    /// bx in event
    int m_totalBxInEvent; 

    /// trigger decision

    /// number of physics trigger algorithms
    unsigned int m_numberPhysTriggers;

    /// trigger objects

    /// muons
    unsigned int m_numberL1Mu;

    /// e/gamma  objects
    unsigned int m_numberL1EG;


    ///  jets
    unsigned int m_numberL1Jet;
    
    ///  taus
    unsigned int m_numberL1Tau;

    /// hardware

    /// number of condition chips
    unsigned int m_numberChips;

    /// number of pins on the GTL condition chips
    unsigned int m_pinsOnChip;

    /// correspondence "condition chip - GTL algorithm word" in the hardware
    std::vector<int> m_orderOfChip;

    int m_version;
    std::vector<int> m_exp_ints;
    std::vector<double> m_exp_doubles;

};

#endif /*CondFormats_L1TObjects_L1TGlobalParameters_h*/
