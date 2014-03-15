#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalSeverityLevel.h"
#include "DataFormats/Common/interface/RefProd.h" 
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/Holder.h"
#include <vector>

namespace DataFormats_EcalRecHit {
  struct dictionary {
    EcalUncalibratedRecHit _ahit;
    std::vector<EcalUncalibratedRecHit> _hitVect;
    edm::SortedCollection<EcalUncalibratedRecHit> _theEURsc;
    EcalUncalibratedRecHitCollection _theEURHitCollection;
    EBUncalibratedRecHitCollection _theEBURHitCollection;
    EEUncalibratedRecHitCollection _theEEURHitCollection;
    edm::Wrapper< EcalUncalibratedRecHitCollection > _EURHitProd;
    edm::Wrapper< EBUncalibratedRecHitCollection > _EBURHitProd;
    edm::Wrapper< EEUncalibratedRecHitCollection > _EEURHitProd;

    EcalSeverityLevel::SeverityLevel sl;

    EcalUncalibratedRecHitRef _EURHitRef;
    EcalUncalibratedRecHitRefs _EURHitRefs;
    EcalUncalibratedRecHitsRef _EURHitsRef;

    EcalRecHit _aRecHit;
    std::vector<EcalRecHit> _ERHitVect;
    edm::SortedCollection<EcalRecHit> _theERsc;
    EcalRecHitCollection _theERHitCollection;
    EBRecHitCollection _theEBRHitCollection;
    EERecHitCollection _theEERHitCollection;
    edm::Wrapper< EcalRecHitCollection > _ERHitProd;
    edm::Wrapper< EBRecHitCollection > _EBRHitProd;
    edm::Wrapper< EERecHitCollection > _EERHitProd;
    EcalRecHitRef _ERHitRef;
    EcalRecHitRefs _ERHitRefs;
    EcalRecHitsRef _ERHitsRef;
  };
}

//raw to rechit specific formats
#include "DataFormats/Common/interface/RefGetter.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitComparison.h"
#include "boost/cstdint.hpp" 

namespace DataFormats_EcalRecHit {
  struct dictionary2 {
    edm::Wrapper< EcalRecHit > dummy01;
    edm::Wrapper< std::vector<EcalRecHit>  > dummy11;
    edm::Wrapper< edm::DetSet<EcalRecHit> > dummy21;
    edm::Wrapper< std::vector<edm::DetSet<EcalRecHit> > > dummy31;
    edm::Wrapper< edm::DetSetVector<EcalRecHit> > dummy41;
    edm::Wrapper< std::vector< std::vector < edm::DetSet<EcalRecHit> > > > dummy51;
    edm::Wrapper< edm::RegionIndex<EcalRecHit> > dummy71;
    edm::Wrapper< std::vector< edm::RegionIndex<EcalRecHit> > > dummy72;
    edm::Wrapper< edm::LazyGetter<EcalRecHit> > dummy73;
    edm::Wrapper< edm::Ref<edm::LazyGetter<EcalRecHit>,edm::RegionIndex<EcalRecHit>,edm::FindRegion<EcalRecHit> > > dummy74;
    edm::Wrapper< std::vector<edm::Ref<edm::LazyGetter<EcalRecHit>,edm::RegionIndex<EcalRecHit>,edm::FindRegion<EcalRecHit> > > > dummy75;
    edm::Wrapper< edm::RefGetter<EcalRecHit> > dummy76;
    edm::Wrapper< edm::Ref< edm::LazyGetter<EcalRecHit>, EcalRecHit, edm::FindValue<EcalRecHit> > > dummy77;
  };
}
