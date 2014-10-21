#ifndef CondFormats_L1TObjects_L1GtPrescaleFactors_h
#define CondFormats_L1TObjects_L1GtPrescaleFactors_h

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

// system include files
#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>
#include <ostream>

// user include files
//   base class

// forward declarations

// class declaration
class L1GtPrescaleFactors
{

public:

    // constructor
    L1GtPrescaleFactors();

    //  from a vector of prescale-factor set
    L1GtPrescaleFactors(const std::vector<std::vector<int> >&);

    // destructor
    virtual ~L1GtPrescaleFactors();

public:

    /// get the prescale factors by reference
    inline const std::vector<std::vector<int> >& gtPrescaleFactors() const
    {
        return m_prescaleFactors;
    }

    /// set the prescale factors
    void setGtPrescaleFactors(const std::vector<std::vector<int> >&);

    /// print the prescale factors
    void print(std::ostream&) const;

private:

    /// prescale factors
    std::vector<std::vector<int> > m_prescaleFactors;


    COND_SERIALIZABLE;
};

#endif /*CondFormats_L1TObjects_L1GtPrescaleFactors_h*/
