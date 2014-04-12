#include <iostream>

#include "CalibCalorimetry/EcalLaserAnalyzer/interface/MEChannel.h"
#include "CalibCalorimetry/EcalLaserAnalyzer/interface/ME.h"

int main(int argc, char **argv)
{
        for( int ireg=0; ireg<ME::iSizeE; ireg++ )
        {
                ME::regTree(ireg)->print(std::cout,true);
        }
        return 0;
}
