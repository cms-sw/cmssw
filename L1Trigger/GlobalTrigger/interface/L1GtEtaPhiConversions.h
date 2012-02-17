#ifndef GlobalTrigger_L1GtEtaPhiConversions_h
#define GlobalTrigger_L1GtEtaPhiConversions_h

/**
 * \class L1GtEtaPhiConversions
 * 
 * 
 * Description: convert eta and phi between various L1 trigger objects.
 * 
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete   - HEPHY Vienna 
 * 
 * $Date$
 * $Revision$
 *
 */

// system include files
#include <iostream>

#include <string>
#include <vector>

#include <boost/cstdint.hpp>

// user include files

//   base class

//
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapFwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// forward declarations
class L1CaloGeometry;
class L1MuTriggerScales;

// class interface
class L1GtEtaPhiConversions
{

public:

    /// constructor
    L1GtEtaPhiConversions();

    /// destructor
    virtual ~L1GtEtaPhiConversions();

public:

    /// perform all conversions
    void convert(const L1CaloGeometry*, const L1MuTriggerScales*, const int,
            const int);

    /// print all the performed conversions
    virtual void print(std::ostream& myCout) const;

private:

    /// phi conversion for Mu to Calo 
    std::vector<unsigned int> m_lutPhiMuCalo; 

    /// eta conversion of calorimeter object to a common scale
    std::vector<unsigned int> m_lutEtaCenCaloCommon; 

};

#endif
