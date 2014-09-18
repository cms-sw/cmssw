#ifndef HcalODFCorrections_h
#define HcalODFCorrections_h

/** 

class HcalODFCorrections

POOL object to store ODF Corrections values
ODF stands for Optical Density Filter

$Author: Audrius Mecionis

eta	phi	low	high
-1	6	0.635	0.875
-2	6	0.623	0.937
-3	6	0.67	0.942
-4	6	0.633	0.9
-5	6	0.644	0.922
-6	6	0.648	0.925
-7	6	0.6	0.901
-8	6	0.57	0.85
-9	6	0.595	0.852
-10	6	0.554	0.818
-11	6	0.505	0.731
-12	6	0.513	0.717
-13	6	0.515	0.782
-14	6	0.561	0.853
-15	6	0.579	0.778
-1	32	0.741	0.973
-2	32	0.721	0.925
-3	32	0.73	0.9
-4	32	0.698	0.897
-5	32	0.708	0.95
-6	32	0.751	0.935
-7	32	0.861	1


*/

#include "CondFormats/Serialization/interface/Serializable.h"
#include <vector>
#include <boost/cstdint.hpp>

class HcalODFCorrections {
    
    public:
        HcalODFCorrections();
        ~HcalODFCorrections();
        
        struct Item {
            int eta;
            int phi;
            float low;
            float high;
          
            COND_SERIALIZABLE;
        };
    
    std::vector<Item> odfCorrections;
    
    COND_SERIALIZABLE;
};

#endif
