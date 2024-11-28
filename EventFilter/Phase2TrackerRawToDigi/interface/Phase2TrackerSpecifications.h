#ifndef PHASE2TRACKERSPECIFICATIONS_H
#define PHASE2TRACKERSPECIFICATIONS_H

#include <bitset>

// Detector Specifications for Phase 2, adapted from 
// https://github.com/cms-L1TK/cmssw/blob/f093f1b30f436c7323f132d7a0f51753ccf0ae3b/EventFilter/Phase2TrackerRawToDigi/interface/utils.h

namespace Phase2TrackerSpecifications
{
    static const int MAX_SSA_PER_PS_MODULE = 16;
    static const int MAX_MPA_PER_PS_MODULE = 16;
    static const int MAX_CBC_PER_2S_MODULE = 16;

    static const int SLINKS_PER_DTC = 4;

    static const int STRIPS_PER_CBC = 127;
    static const int CHANNELS_PER_CBC = 254;

    static const int STRIPS_PER_SSA = 120;
    static const int CHANNELS_PER_SSA = 240;

    static const int MODULES_PER_SLINK = 18;
    static const int CICs_PER_SLINK = 36;

    static const int CIC_Z_BOUNDARY_PIXEL = 15;
    static const int CIC_Z_BOUNDARY_STRIPS = 0;

    static const int MAX_DTC_ID = 216;
    static const int MIN_DTC_ID = 1;

    static const int MIN_SLINK_ID = 0;
    static const int MAX_SLINK_ID = 3;

    static const int TRACKER_HEADER = 0;

};

#endif