#ifndef PHASE2TRACKERSPECIFICATIONS_H
#define PHASE2TRACKERSPECIFICATIONS_H

// Detector Specifications for Phase 2, adapted from 
// https://github.com/cms-L1TK/cmssw/blob/f093f1b30f436c7323f132d7a0f51753ccf0ae3b/EventFilter/Phase2TrackerRawToDigi/interface/utils.h

namespace Phase2TrackerSpecifications
{
    static const int MAX_SSA_PER_PS_MODULE = 16;
    static const int MAX_MPA_PER_PS_MODULE = 16;
    static const int MAX_CBC_PER_2S_MODULE = 16;

    static const int STRIPS_PER_CBC = 127;
    static const int CHANNELS_PER_CBC = 254;

    static const int STRIPS_PER_SSA = 120;
    static const int CHANNELS_PER_SSA = 240;

    static const int MODULES_PER_SLINK = 18;
    static const int CICs_PER_SLINK = 36;

};

#endif