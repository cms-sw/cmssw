#ifndef TrackerRecHit2D_CLASSES_H
#define TrackerRecHit2D_CLASSES_H

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2DCollection.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include <vector>

namespace {
  namespace {    
    SiStripRecHit2D a1;
    SiStripMatchedRecHit2D a2;
    SiPixelRecHit b1;
    SiTrackerGSRecHit2D c1;
    DetId a3;

    edm::ClonePolicy<SiStripRecHit2D>  a4;
    edm::ClonePolicy<SiStripMatchedRecHit2D > a5;
    edm::ClonePolicy<SiPixelRecHit> b2;
    edm::ClonePolicy<SiTrackerGSRecHit2D>  c2;

    edm::OwnVector<SiStripRecHit2D,
      edm::ClonePolicy<SiStripRecHit2D> > a6;
    edm::OwnVector<SiStripRecHit2D,
      edm::ClonePolicy<SiStripRecHit2D> >::const_iterator it6;
    edm::OwnVector<SiStripMatchedRecHit2D,
      edm::ClonePolicy<SiStripMatchedRecHit2D> > a7;
    edm::OwnVector<SiStripMatchedRecHit2D,
      edm::ClonePolicy<SiStripMatchedRecHit2D> >::const_iterator it7;
    edm::OwnVector<SiPixelRecHit,
      edm::ClonePolicy<SiPixelRecHit> > b3;
    edm::OwnVector<SiPixelRecHit,
      edm::ClonePolicy<SiPixelRecHit> >::const_iterator it3;
    edm::OwnVector<SiTrackerGSRecHit2D,
      edm::ClonePolicy<SiTrackerGSRecHit2D> > c3;
    edm::OwnVector<SiTrackerGSRecHit2D,
      edm::ClonePolicy<SiTrackerGSRecHit2D> >::const_iterator it8;
    
    edm::Wrapper< edm::RangeMap<DetId,
      edm::OwnVector<SiStripRecHit2D,
      edm::ClonePolicy<SiStripRecHit2D> >, 
      edm::ClonePolicy<SiStripRecHit2D> > >    siStripRecHit2DLocalPosCollectionWrapper;
    edm::RangeMap<DetId,
      edm::OwnVector<SiStripRecHit2D,
      edm::ClonePolicy<SiStripRecHit2D> >, 
      edm::ClonePolicy<SiStripRecHit2D> >::id_iterator    it2d;
    
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
      edm::OwnVector<SiTrackerGSRecHit2D,
      edm::ClonePolicy<SiTrackerGSRecHit2D> >, 
      edm::ClonePolicy<SiTrackerGSRecHit2D> > >    siStripGaussianSmearingRecHit2DLocalPosCollectionWrapper;
    edm::RangeMap<DetId,
      edm::OwnVector<SiTrackerGSRecHit2D,
      edm::ClonePolicy<SiTrackerGSRecHit2D> >, 
      edm::ClonePolicy<SiTrackerGSRecHit2D> >::id_iterator    itgs2d;
    
    edm::Ref<edm::DetSetVector<SiPixelCluster>, 
      SiPixelCluster, 
      edm::refhelper::FindForDetSetVector<SiPixelCluster> > siPixelClusterReference;

    edm::Ref<edm::DetSetVector<SiStripCluster>, 
      SiStripCluster, 
      edm::refhelper::FindForDetSetVector<SiStripCluster> > siStripClusterReference;

    edm::RefBase<std::pair<unsigned int,unsigned int> > p1;
    edm::RefItem<std::pair<unsigned int,unsigned int> > p2;

  }
}

#endif // SISTRIPRECHIT_CLASSES_H
