#ifndef DPGAnalysis_SiStripTools_classes_H
#define DPGAnalysis_SiStripTools_classes_H

#include "DPGAnalysis/SiStripTools/interface/TinyEvent.h"
#include "DPGAnalysis/SiStripTools/interface/EventWithHistory.h"
#include "DPGAnalysis/SiStripTools/interface/APVCyclePhaseCollection.h"
#include "DPGAnalysis/SiStripTools/interface/Multiplicities.h"

#include "DataFormats/Common/interface/Wrapper.h"

namespace { 
  struct dictionary {
    TinyEventCollection dummycoll;
    SingleSiStripDigiMultiplicity dummy1;
    SingleSiPixelClusterMultiplicity dummy2;
    SingleSiStripClusterMultiplicity dummy3;
    SiPixelClusterSiStripClusterMultiplicityPair dummy4;
    ClusterSummarySingleMultiplicity dummy5;
    ClusterSummaryMultiplicityPair dummy6;

    edm::Wrapper<TinyEventCollection> dummywrappedcoll;
    edm::Wrapper<EventWithHistory>  dummywrappedEWH;
    edm::Wrapper<APVCyclePhaseCollection> dummywrappedAPVC;

    /*
    edm::Wrapper<SingleSiStripDigiMultiplicity> dummywrapped1;
    edm::Wrapper<SingleSiStripClusterMultiplicity> dummywrapped2;
    edm::Wrapper<SingleSiPixelClusterMultiplicity> dummywrapped3;
    edm::Wrapper<SiPixelClusterSiStripClusterMultiplicityPair> dummywrapped4;
    */
  };
}

#endif // DPGAnalysis_SiStripTools_classes_H
