#ifndef TrackerRecHit2D_CLASSES_H
#define TrackerRecHit2D_CLASSES_H

#include "DataFormats/TrackerRecHit2D/interface/OmniClusterRef.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1DCollection.h"
#include "DataFormats/CLHEP/interface/Migration.h" 
#include <boost/cstdint.hpp> 
#include "DataFormats/GeometryVector/interface/LocalPoint.h" 
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h" 
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/Common/interface/RefProd.h" 
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h" 
#include "DataFormats/Common/interface/DetSetVector.h" 
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RangeMap.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/ClusterRemovalInfo.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerMultiRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/FastTrackerRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/FastMatchedTrackerRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/FastProjectedTrackerRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/FastTrackerRecHitCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/Phase2TrackerRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/MTDTrackingRecHit.h"
#include <vector>

namespace DataFormats_TrackerRecHit2D {
  struct dictionary {
    ProjectedSiStripRecHit2D projHit;   
    SiStripRecHit2D a1;
    SiStripRecHit1D a11;
    SiStripMatchedRecHit2D a2;
    SiPixelRecHit b1;
    MTDTrackingRecHit mtd1;

    edm::ClonePolicy<SiStripRecHit2D>  a4;
    edm::ClonePolicy<SiStripRecHit1D>  a44;
    edm::ClonePolicy<SiStripMatchedRecHit2D > a5;
    edm::ClonePolicy<SiPixelRecHit> b2;
    edm::ClonePolicy<SiTrackerMultiRecHit>  e2;
    edm::ClonePolicy<MTDTrackingRecHit>  e10;

    edm::OwnVector<SiStripRecHit2D,
      edm::ClonePolicy<SiStripRecHit2D> > a6;
    edm::OwnVector<SiStripRecHit2D,
      edm::ClonePolicy<SiStripRecHit2D> >::const_iterator it6;
    edm::OwnVector<SiStripRecHit1D,
      edm::ClonePolicy<SiStripRecHit1D> > a66;
    edm::OwnVector<SiStripRecHit1D,
      edm::ClonePolicy<SiStripRecHit1D> >::const_iterator it66;
    edm::OwnVector<SiStripMatchedRecHit2D,
      edm::ClonePolicy<SiStripMatchedRecHit2D> > a7;
    edm::OwnVector<SiStripMatchedRecHit2D,
      edm::ClonePolicy<SiStripMatchedRecHit2D> >::const_iterator it7;
    edm::OwnVector<SiPixelRecHit,
      edm::ClonePolicy<SiPixelRecHit> > b3;
    edm::OwnVector<SiPixelRecHit,
      edm::ClonePolicy<SiPixelRecHit> >::const_iterator it3;
    edm::OwnVector<SiTrackerMultiRecHit,
      edm::ClonePolicy<SiTrackerMultiRecHit> > e3;
    edm::OwnVector<SiTrackerMultiRecHit,
      edm::ClonePolicy<SiTrackerMultiRecHit> >::const_iterator it10;
    MTDTrackingOwnVector::const_iterator it11;

    edm::OwnVector<BaseTrackerRecHit> ovbtrh;
    edm::Wrapper<edm::OwnVector<BaseTrackerRecHit>> wovbtrh;

    edm::Wrapper< edm::RangeMap<DetId,
      edm::OwnVector<SiStripRecHit2D,
      edm::ClonePolicy<SiStripRecHit2D> >, 
      edm::ClonePolicy<SiStripRecHit2D> > >    siStripRecHit2DLocalPosCollectionWrapper;
    edm::RangeMap<DetId,
      edm::OwnVector<SiStripRecHit2D,
      edm::ClonePolicy<SiStripRecHit2D> >, 
      edm::ClonePolicy<SiStripRecHit2D> >::id_iterator    it2d;

    edm::Wrapper< edm::RangeMap<DetId,
      edm::OwnVector<SiStripRecHit1D,
      edm::ClonePolicy<SiStripRecHit1D> >, 
      edm::ClonePolicy<SiStripRecHit1D> > >    siStripRecHit1DLocalPosCollectionWrapper;
    edm::RangeMap<DetId,
      edm::OwnVector<SiStripRecHit1D,
      edm::ClonePolicy<SiStripRecHit1D> >, 
      edm::ClonePolicy<SiStripRecHit1D> >::id_iterator    it1d;

    
    edm::Wrapper< edm::RangeMap<DetId,
      edm::OwnVector<SiStripMatchedRecHit2D,
      edm::ClonePolicy<SiStripMatchedRecHit2D> >, 
      edm::ClonePolicy<SiStripMatchedRecHit2D> > > siStripRecHit2DMatchedLocalPosCollectionWrapper;
    edm::RangeMap<DetId,
      edm::OwnVector<SiStripMatchedRecHit2D,
      edm::ClonePolicy<SiStripMatchedRecHit2D> >, 
      edm::ClonePolicy<SiStripMatchedRecHit2D> >::id_iterator itmatch;

    edm::Wrapper< edm::RangeMap<DetId,
      edm::OwnVector<SiPixelRecHit,
      edm::ClonePolicy<SiPixelRecHit> >, 
      edm::ClonePolicy<SiPixelRecHit> > >  siPixelRecHitCollectionWrapper;
    edm::RangeMap<DetId,
      edm::OwnVector<SiPixelRecHit,
      edm::ClonePolicy<SiPixelRecHit> >, 
      edm::ClonePolicy<SiPixelRecHit> >::id_iterator itpix;

    edm::Wrapper< edm::RangeMap<DetId,
      edm::OwnVector<MTDTrackingRecHit,
      edm::ClonePolicy<MTDTrackingRecHit> >, 
      edm::ClonePolicy<MTDTrackingRecHit> > >  mtdRecHitCollectionWrapper;
    edm::RangeMap<DetId,
      edm::OwnVector<MTDTrackingRecHit,
      edm::ClonePolicy<MTDTrackingRecHit> >, 
      edm::ClonePolicy<MTDTrackingRecHit> >::id_iterator mtdpix;

    edm::Ref<edm::RangeMap<DetId,edm::OwnVector<SiStripRecHit2D,edm::ClonePolicy<SiStripRecHit2D> >,edm::ClonePolicy<SiStripRecHit2D> >,SiStripRecHit2D,edm::refhelper::FindUsingAdvance<edm::RangeMap<DetId,edm::OwnVector<SiStripRecHit2D,edm::ClonePolicy<SiStripRecHit2D> >,edm::ClonePolicy<SiStripRecHit2D> >,SiStripRecHit2D> > refRangeMapDetIdOwnVectorSiStripRecHit2D;
    edm::RefVector<edm::RangeMap<DetId,edm::OwnVector<SiStripRecHit2D,edm::ClonePolicy<SiStripRecHit2D> >,edm::ClonePolicy<SiStripRecHit2D> >,SiStripRecHit2D,edm::refhelper::FindUsingAdvance<edm::RangeMap<DetId,edm::OwnVector<SiStripRecHit2D,edm::ClonePolicy<SiStripRecHit2D> >,edm::ClonePolicy<SiStripRecHit2D> >,SiStripRecHit2D> > refVectorRangeMapDetIdOwnVectorSiStripRecHit2D;

    edm::Ref<edm::RangeMap<DetId,edm::OwnVector<SiStripRecHit1D,edm::ClonePolicy<SiStripRecHit1D> >,edm::ClonePolicy<SiStripRecHit1D> >,SiStripRecHit1D,edm::refhelper::FindUsingAdvance<edm::RangeMap<DetId,edm::OwnVector<SiStripRecHit1D,edm::ClonePolicy<SiStripRecHit1D> >,edm::ClonePolicy<SiStripRecHit1D> >,SiStripRecHit1D> > refRangeMapDetIdOwnVectorSiStripRecHit1D;
    edm::RefVector<edm::RangeMap<DetId,edm::OwnVector<SiStripRecHit1D,edm::ClonePolicy<SiStripRecHit1D> >,edm::ClonePolicy<SiStripRecHit1D> >,SiStripRecHit1D,edm::refhelper::FindUsingAdvance<edm::RangeMap<DetId,edm::OwnVector<SiStripRecHit1D,edm::ClonePolicy<SiStripRecHit1D> >,edm::ClonePolicy<SiStripRecHit1D> >,SiStripRecHit1D> > refVectorRangeMapDetIdOwnVectorSiStripRecHit1D;


    edm::Wrapper<edmNew::DetSetVector<SiStripRecHit2D> > wdstvDummy1;
    edm::Wrapper<edmNew::DetSetVector<SiStripRecHit1D> > wdstvDummy11;
    edm::Wrapper<edmNew::DetSetVector<SiStripMatchedRecHit2D> > wdstvDummy2;
    edm::Wrapper<edmNew::DetSetVector<SiPixelRecHit> > wdstvDummy3;
    edm::Wrapper<MTDTrackingDetSetVector> wdstvDummy4;

    edm::Wrapper<reco::ClusterRemovalInfo> clusterRemovalInfo;

      edm::OwnVector<FastTrackerRecHit,edm::ClonePolicy<FastTrackerRecHit> > fastsimTrackerRecHitCollection;
      edm::Wrapper<edm::OwnVector<FastTrackerRecHit,edm::ClonePolicy<FastTrackerRecHit> > > fastsimTrackerRecHitCollection_Wrapper;

      std::vector<edm::Ref<edm::OwnVector<FastTrackerRecHit,edm::ClonePolicy<FastTrackerRecHit> >,FastTrackerRecHit,edm::refhelper::FindUsingAdvance<edm::OwnVector<FastTrackerRecHit,edm::ClonePolicy<FastTrackerRecHit> >,FastTrackerRecHit> > > fastsimTrackerRecHitRefCollection;

      edm::Wrapper<std::vector<edm::Ref<edm::OwnVector<FastTrackerRecHit,edm::ClonePolicy<FastTrackerRecHit> >,FastTrackerRecHit,edm::refhelper::FindUsingAdvance<edm::OwnVector<FastTrackerRecHit,edm::ClonePolicy<FastTrackerRecHit> >,FastTrackerRecHit> > > > fastsimTrackerRecHitRefCollection_Wrapper;

      std::vector<edm::OwnVector<FastTrackerRecHit,edm::ClonePolicy<FastTrackerRecHit> > > fastsimTrackerRecHitCombinations;
      edm::Wrapper<std::vector<edm::OwnVector<FastTrackerRecHit,edm::ClonePolicy<FastTrackerRecHit> > > >fastsimTrackerRecHitCombinations_Wrapper;

      std::vector<std::vector<edm::Ref<edm::OwnVector<FastTrackerRecHit,edm::ClonePolicy<FastTrackerRecHit> >,FastTrackerRecHit,edm::refhelper::FindUsingAdvance<edm::OwnVector<FastTrackerRecHit,edm::ClonePolicy<FastTrackerRecHit> >,FastTrackerRecHit> > > > fastSimTrackerRecHitCombinationCollection;
      edm::Wrapper<std::vector<std::vector<edm::Ref<edm::OwnVector<FastTrackerRecHit,edm::ClonePolicy<FastTrackerRecHit> >,FastTrackerRecHit,edm::refhelper::FindUsingAdvance<edm::OwnVector<FastTrackerRecHit,edm::ClonePolicy<FastTrackerRecHit> >,FastTrackerRecHit> > > > > fastSimTrackerRecHitCombinationCollection_Wrapper;

    edm::Ref<std::vector<std::vector<edm::Ref<edm::OwnVector<FastTrackerRecHit,edm::ClonePolicy<FastTrackerRecHit> >,FastTrackerRecHit,edm::refhelper::FindUsingAdvance<edm::OwnVector<FastTrackerRecHit,edm::ClonePolicy<FastTrackerRecHit> >,FastTrackerRecHit> > > >,std::vector<edm::Ref<edm::OwnVector<FastTrackerRecHit,edm::ClonePolicy<FastTrackerRecHit> >,FastTrackerRecHit,edm::refhelper::FindUsingAdvance<edm::OwnVector<FastTrackerRecHit,edm::ClonePolicy<FastTrackerRecHit> >,FastTrackerRecHit> > >,edm::refhelper::FindUsingAdvance<std::vector<std::vector<edm::Ref<edm::OwnVector<FastTrackerRecHit,edm::ClonePolicy<FastTrackerRecHit> >,FastTrackerRecHit,edm::refhelper::FindUsingAdvance<edm::OwnVector<FastTrackerRecHit,edm::ClonePolicy<FastTrackerRecHit> >,FastTrackerRecHit> > > >,std::vector<edm::Ref<edm::OwnVector<FastTrackerRecHit,edm::ClonePolicy<FastTrackerRecHit> >,FastTrackerRecHit,edm::refhelper::FindUsingAdvance<edm::OwnVector<FastTrackerRecHit,edm::ClonePolicy<FastTrackerRecHit> >,FastTrackerRecHit> > > > > fastSimTrackerRecHitCombinationRef;

        edm::Wrapper< Phase2TrackerRecHit1D > cl0;
        edm::Wrapper< std::vector< Phase2TrackerRecHit1D > > cl1;
        edm::Wrapper< edmNew::DetSet< Phase2TrackerRecHit1D > > cl2;
        edm::Wrapper< std::vector< edmNew::DetSet< Phase2TrackerRecHit1D > > > cl3;
        edm::Wrapper< Phase2TrackerRecHit1DCollectionNew > cl4;

  };
}

#endif // SISTRIPRECHIT_CLASSES_H
