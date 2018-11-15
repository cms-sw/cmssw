#include "DataFormats/FTLRecHit/interface/FTLUncalibratedRecHit.h"
#include "DataFormats/FTLRecHit/interface/FTLRecHit.h"
#include "DataFormats/FTLRecHit/interface/FTLRecHitCollections.h"
#include "DataFormats/FTLRecHit/interface/FTLSeverityLevel.h"
#include "DataFormats/FTLRecHit/interface/FTLTrackingRecHit.h"
#include "DataFormats/Common/interface/RefProd.h" 
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/Holder.h"
#include <vector>

namespace DataFormats_FTLRecHit {
  struct dictionary {
    FTLUncalibratedRecHit _ahit;
    std::vector<FTLUncalibratedRecHit> _hitVect;
    edm::SortedCollection<FTLUncalibratedRecHit> _theFTLURsc;
    edm::Wrapper< FTLUncalibratedRecHitCollection > _FTLURHitProd;

    FTLSeverityLevel::SeverityLevel sl;

    FTLUncalibratedRecHitRef _FTLURHitRef;
    FTLUncalibratedRecHitRefs _FTLURHitRefs;
    FTLUncalibratedRecHitsRef _FTLURHitsRef;

    FTLRecHit _aRecHit;
    std::vector<FTLRecHit> _FTLRHitVect;
    edm::SortedCollection<FTLRecHit> _theFTLRsc;

    edm::Wrapper< FTLRecHitCollection > _FTLRHitProd;
    FTLRecHitRef _FTLRHitRef;
    FTLRecHitRefs _FTLRHitRefs;
    FTLRecHitsRef _FTLRHitsRef;
    
    FTLTrackingRecHitFromHit _FTLTTrackingRecHitFromRef;
    FTLTrackingRecHitCollection _FTLTTrackingRecHitCollection;
    edm::Wrapper<FTLTrackingRecHitCollection> _FTLTTrackingRecHitProd;
    
    
  };
}

//raw to rechit specific formats
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/FTLRecHit/interface/FTLRecHitComparison.h"
#include "boost/cstdint.hpp" 

namespace DataFormats_FTLRecHit {
  struct dictionary2 {
    edm::Wrapper< FTLRecHit > dummy01;
    edm::Wrapper< std::vector<FTLRecHit>  > dummy11;
    edm::Wrapper< edm::DetSet<FTLRecHit> > dummy21;
    edm::Wrapper< std::vector<edm::DetSet<FTLRecHit> > > dummy31;
    edm::Wrapper< edm::DetSetVector<FTLRecHit> > dummy41;
    edm::Wrapper< std::vector< std::vector < edm::DetSet<FTLRecHit> > > > dummy51;
  };
}
