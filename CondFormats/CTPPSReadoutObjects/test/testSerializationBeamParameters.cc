#include "CondFormats/Serialization/interface/Test.h"
#include "CondFormats/CTPPSReadoutObjects/src/headers.h"

int main()
{
    testSerialization<CTPPSBeamParameters>() ;
    
    return 0 ;
}
