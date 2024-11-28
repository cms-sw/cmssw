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

        DTCAssembly(unsigned int& event) : dtcUnits_(216, DTCUnit(event)), eventId_(event) {}

        /** Get Methods **/
        DTCUnit& GetDTCUnit(const unsigned int& DTCUnitID) 
        { 
            // if DTCUnitID is out of range [0, 216], throw cms exception
            if (DTCUnitID > Phase2TrackerSpecifications::MAX_DTC_ID) 
            {
                throw cms::Exception("DTCAssembly") << "DTCUnitID " << DTCUnitID << " is out of range [1, 216]";
            }
            else
            {
                return dtcUnits_[DTCUnitID - 1]; 
            }
        }

        std::vector<DTCUnit>& GetDTCUnits() { return dtcUnits_; }

    private:

        std::vector<DTCUnit> dtcUnits_;
        unsigned int eventId_;

};

typedef std::vector<DTCAssembly> DTCAssemblyCollection;
typedef edm::Ref<DTCAssemblyCollection> DTCAssemblyRef;

#endif