#ifndef DTCASSEMBLY_H
#define DTCASSEMBLY_H

#include <vector>
#include "FWCore/Utilities/interface/Exception.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerSpecifications.h"
#include "./DTCUnit.h"

/**
 * @class DTCAssembly
 * @brief Class to represent a collection of DTCs for the phase 2 tracker,
 * a collection of interface/DTCUnit.h objects
 */

class DTCAssembly
{

    public:

        DTCAssembly(unsigned int& event) : DTCUnits(216, DTCUnit(event)), EventID(event) {}

        /** Get Methods **/
        DTCUnit& GetDTCUnit(const unsigned int& DTCUnitID) 
        { 
            // if DTCUnitID is out of range [0, 216], throw cms exception
            if (DTCUnitID > Phase2TrackerSpecifications::MAX_DTCs) 
            {
                throw cms::Exception("DTCAssembly") << "DTCUnitID " << DTCUnitID << " is out of range [1, 216]";
            }
            else
            {
                return DTCUnits[DTCUnitID - 1]; 
            }
        }

        std::vector<DTCUnit>& GetDTCUnits() { return DTCUnits; }

        /** Other Methods **/
        // void Clear() { for (auto& element : DTCUnits) { element = DTCUnit(); } }

    private:

        std::vector<DTCUnit> DTCUnits;
        unsigned int EventID;

};

typedef std::vector<DTCAssembly> DTCAssemblyCollection;
typedef edm::Ref<DTCAssemblyCollection> DTCAssemblyRef;

#endif