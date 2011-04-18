#ifndef DPGAnalysis_SiStripTools_classes_H
#define DPGAnalysis_SiStripTools_classes_H

#include "DPGAnalysis/SiStripTools/interface/TinyEvent.h"
#include "DPGAnalysis/SiStripTools/interface/EventWithHistory.h"
#include "DPGAnalysis/SiStripTools/interface/APVCyclePhaseCollection.h"

#include "DataFormats/Common/interface/Wrapper.h"

namespace { 
  struct dictionary {
    TinyEventCollection dummycoll;
    edm::Wrapper<TinyEventCollection> dummywrappedcoll;
    edm::Wrapper<EventWithHistory>  dummywrappedEWH;
    edm::Wrapper<APVCyclePhaseCollection> dummywrappedAPVC;
  };
}

#endif // DPGAnalysis_SiStripTools_classes_H
