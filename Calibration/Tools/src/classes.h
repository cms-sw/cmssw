#include "Calibration/Tools/interface/IC.h"
#include "Calibration/Tools/interface/DRings.h"

namespace {
        struct dictionary {
                IC ic;
                DSAll all;
                DSIsNextToBoundaryEB isNextToBoundaryEB;
                DSIsNextToProblematicEB isNextToProblematicEB;
                DSIsNextToProblematicEE isNextToProblematicEE;
                DSIsNextToProblematicEEPlus isNextToProblematicEEPlus;
                DSIsNextToProblematicEEMinus isNextToProblematicEEMinus;
                DSIsEndcap isEndcap;
                DSIsEndcapPlus isEndcapPlus;
                DSIsEndcapMinus isEndcapMinus;
                DSIsBarrel isBarrel;
                DSHasChannelStatusEB chStEB;
                DSHasChannelStatusEE chStEE;
                DRings rings;
        };
}
