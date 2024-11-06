#ifndef DTCASSEMBLY_H
#define DTCASSEMBLY_H

#include <vector>
#include "FWCore/Utilities/interface/Exception.h"
#include "./DTCUnit.h"

class DTCAssembly
{

    public:

        DTCAssembly(unsigned int& event) : DTCUnits(216, DTCUnit(event)), EventID(event) {}

        /** Get Methods **/
        DTCUnit& GetDTCUnit(const unsigned int& DTCUnitID) 
        { 
            // if DTCUnitID is out of range [0, 216], throw cms exception
            if (DTCUnitID > 216) 
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