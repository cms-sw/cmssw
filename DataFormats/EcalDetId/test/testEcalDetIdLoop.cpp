/**
   \file
   Test loop on EcalDetId
*/

#include <DataFormats/EcalDetId/interface/EBDetId.h>
#include <DataFormats/EcalDetId/interface/EEDetId.h>

#include <iostream>

int main()
{
        std::cout << "Example of loop on DetId" << std::endl;
        int cntEB = 0;
        int cntEE = 0;
        // Barrel
        std::cerr << "BARREL --" << std::endl;
        for ( int i = 0; i < EBDetId::kSizeForDenseIndexing; ++i )
        {
                EBDetId id = EBDetId::unhashIndex( i );
                if ( id != EBDetId(0) ) {
                        std::cout << " id " << id.rawId() << " -> (" << id.ieta() << ", " << id.iphi() << ") " << id.ic() << "\n";
                }
                ++cntEB;
        }
        // Endcap
        std::cerr << "ENDCAP --" << std::endl;
        for ( int i = 0; i < EEDetId::kSizeForDenseIndexing; ++i )
        {
                EEDetId id = EEDetId::unhashIndex( i );
                if ( id != EEDetId(0) ) {
                        std::cout << " id " << id.rawId() << " -> (" << id.ix() << ", " << id.iy() << ", " << id.zside() << ")\n";
                }
                ++cntEE;
        }
        std::cerr << "Total of " << cntEB << " DetId's counted for EB and " << cntEE << " for EE.\n";
        return 0;
}
