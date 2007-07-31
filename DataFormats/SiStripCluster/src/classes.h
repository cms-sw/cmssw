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
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
namespace {
  namespace {
    edm::Wrapper< SiStripCluster > dummy0;
    edm::Wrapper< std::vector<SiStripCluster>  > dummy1;
    edm::Wrapper< edm::DetSet<SiStripCluster> > dummy2;
    edm::Wrapper< std::vector<edm::DetSet<SiStripCluster> > > dummy3;
    edm::Wrapper< edm::DetSetVector<SiStripCluster> > dummy4;
    edm::Wrapper< std::vector< std::vector < edm::DetSet<SiStripCluster> > > > dummy5;
  }
}

#include "boost/cstdint.hpp" 
#include "DataFormats/SiStripCommon/interface/SiStripRefGetter.h"
namespace {
  namespace {
    edm::Wrapper< edm::SiStripLazyGetter<SiStripCluster> > dummy6;
    edm::Wrapper< edm::SiStripRefGetter<SiStripCluster> > dummy7;
  }
}

#endif // SISTRIPCLUSTER_CLASSES_H
