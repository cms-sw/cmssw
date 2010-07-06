#ifndef DPGAnalysis_SiStripTools_classes_H
#define DPGAnalysis_SiStripTools_classes_H

#include "DPGAnalysis/SiStripTools/interface/TinyEvent.h"
#include "DPGAnalysis/SiStripTools/interface/EventWithHistory.h"
#include "DPGAnalysis/SiStripTools/interface/APVCyclePhaseCollection.h"
// For APVLatency object
#include "DPGAnalysis/SiStripTools/interface/APVLatency.h"

//#include "DataFormats/Common/interface/Wrapper.h"

namespace { 
  struct dictionary {
    TinyEventCollection dummycoll;
  };
}

#endif // DPGAnalysis_SiStripTools_classes_H
