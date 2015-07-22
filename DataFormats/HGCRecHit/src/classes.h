#include "DataFormats/HGCRecHit/interface/HGCUncalibratedRecHit.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHit.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/HGCRecHit/interface/HGCSeverityLevel.h"
#include "DataFormats/Common/interface/RefProd.h" 
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/Holder.h"
#include <vector>

namespace DataFormats_HGCalRecHit {
  struct dictionary {
    HGCUncalibratedRecHit _ahit;
    std::vector<HGCUncalibratedRecHit> _hitVect;
    edm::SortedCollection<HGCUncalibratedRecHit> _theHGCURsc;
    HGCeeUncalibratedRecHitCollection _theHGCeeURHitCollection;
    HGChefUncalibratedRecHitCollection _theHGChefURHitCollection;
    HGChebUncalibratedRecHitCollection _theHGChebURHitCollection;
    edm::Wrapper< HGCeeUncalibratedRecHitCollection > _HGCeeURHitProd;
    edm::Wrapper< HGChefUncalibratedRecHitCollection > _HGChefURHitProd;
    edm::Wrapper< HGChebUncalibratedRecHitCollection > _HGChebURHitProd;

    HGCSeverityLevel::SeverityLevel sl;

    HGCUncalibratedRecHitRef _HGCURHitRef;
    HGCUncalibratedRecHitRefs _HGCURHitRefs;
    HGCUncalibratedRecHitsRef _HGCURHitsRef;

    HGCRecHit _aRecHit;
    std::vector<HGCRecHit> _HGCRHitVect;
    edm::SortedCollection<HGCRecHit> _theHGCRsc;
    HGCeeRecHitCollection _theHGCeeRHitCollection;
    HGChefRecHitCollection _theHGChefRHitCollection;
    HGChebRecHitCollection _theHGChebRHitCollection;
    edm::Wrapper< HGCeeRecHitCollection > _HGCeeRHitProd;
    edm::Wrapper< HGChefRecHitCollection > _HGChefRHitProd;
    edm::Wrapper< HGChebRecHitCollection > _HGChebRHitProd;
    HGCRecHitRef _HGCRHitRef;
    HGCRecHitRefs _HGCRHitRefs;
    HGCRecHitsRef _HGCRHitsRef;
    edm::reftobase::Holder<CaloRecHit, HGCRecHitRef> rb6;
  };
}

//raw to rechit specific formats
#include "DataFormats/Common/interface/RefGetter.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitComparison.h"
#include "boost/cstdint.hpp" 

namespace DataFormats_HGCalRecHit {
  struct dictionary2 {
    edm::Wrapper< HGCRecHit > dummy01;
    edm::Wrapper< std::vector<HGCRecHit>  > dummy11;
    edm::Wrapper< edm::DetSet<HGCRecHit> > dummy21;
    edm::Wrapper< std::vector<edm::DetSet<HGCRecHit> > > dummy31;
    edm::Wrapper< edm::DetSetVector<HGCRecHit> > dummy41;
    edm::Wrapper< std::vector< std::vector < edm::DetSet<HGCRecHit> > > > dummy51;
    edm::Wrapper< edm::RegionIndex<HGCRecHit> > dummy71;
    edm::Wrapper< std::vector< edm::RegionIndex<HGCRecHit> > > dummy72;
    edm::Wrapper< edm::LazyGetter<HGCRecHit> > dummy73;
    edm::Wrapper< edm::Ref<edm::LazyGetter<HGCRecHit>,edm::RegionIndex<HGCRecHit>,edm::FindRegion<HGCRecHit> > > dummy74;
    edm::Wrapper< std::vector<edm::Ref<edm::LazyGetter<HGCRecHit>,edm::RegionIndex<HGCRecHit>,edm::FindRegion<HGCRecHit> > > > dummy75;
    edm::Wrapper< edm::RefGetter<HGCRecHit> > dummy76;
    edm::Wrapper< edm::Ref< edm::LazyGetter<HGCRecHit>, HGCRecHit, edm::FindValue<HGCRecHit> > > dummy77;
  };
}
