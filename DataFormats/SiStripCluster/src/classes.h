#ifndef SISTRIPCLUSTER_CLASSES_H
#define SISTRIPCLUSTER_CLASSES_H

#include "DataFormats/SiStripCluster/interface/SiStripClusterCollection.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include <vector>

namespace {
  namespace {
    edm::Wrapper<SiStripClusterCollection> siStripClusterCollectionWrapper;
  }
}

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
namespace {
  namespace {
    edm::Wrapper< SiStripCluster > dummy0;
    edm::Wrapper< std::vector<SiStripCluster>  > dummy1;
    edm::Wrapper< edm::DetSet<SiStripCluster> > dummy2;
    edm::Wrapper< std::vector<edm::DetSet<SiStripCluster> > > dummy3;
    edm::Wrapper< edm::DetSetVector<SiStripCluster> > dummy4;
    edm::Wrapper< std::vector< std::vector < edm::DetSet<SiStripCluster> > > > dummy5;

    edm::Wrapper< edmNew::DetSetVector<SiStripCluster> > dummy4_bis;

    edm::Ref<   edm::DetSetVector<SiStripCluster>, SiStripCluster, edm::refhelper::FindForDetSetVector<   SiStripCluster> >    refOld;
    edm::Ref<edmNew::DetSetVector<SiStripCluster>, SiStripCluster, edmNew::DetSetVector<SiStripCluster>::FindForDetSetVector > refNew;
  }
}

#include "boost/cstdint.hpp" 
#include "DataFormats/SiStripCommon/interface/SiStripRefGetter.h"
namespace {
  namespace {

    edm::Wrapper< edm::RegionIndex<SiStripCluster> > dummy7;
    edm::Wrapper< std::vector< edm::RegionIndex<SiStripCluster> > > dummy8;
    edm::Wrapper< edm::SiStripLazyGetter<SiStripCluster> > dummy9;
    edm::Wrapper< edm::Ref<edm::SiStripLazyGetter<SiStripCluster>,edm::RegionIndex<SiStripCluster>,edm::FindRegion<SiStripCluster> > > dummy10;
    edm::Wrapper< std::vector<edm::Ref<edm::SiStripLazyGetter<SiStripCluster>,edm::RegionIndex<SiStripCluster>,edm::FindRegion<SiStripCluster> > > > dummy12;
    edm::Wrapper< edm::SiStripRefGetter<SiStripCluster> > dummy13;
    edm::Wrapper< edm::Ref< edm::SiStripLazyGetter<SiStripCluster>, SiStripCluster, edm::FindValue<SiStripCluster> > > dummy14;
  }
}

#endif // SISTRIPCLUSTER_CLASSES_H
