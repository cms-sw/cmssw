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
    //
    edm::Wrapper< std::map<unsigned int, std::vector<SiStripCluster> > > dummy5;
  }
}

#include "DataFormats/SiStripCluster/interface/SiStripClusterInfo.h"
namespace {
  namespace {
    edm::Wrapper< SiStripClusterInfo > adummy0;
    edm::Wrapper< std::vector<SiStripClusterInfo>  > adummy1;
    edm::Wrapper< edm::DetSet<SiStripClusterInfo> > adummy2;
    edm::Wrapper< std::vector<edm::DetSet<SiStripClusterInfo> > > adummy3;
    edm::Wrapper< edm::DetSetVector<SiStripClusterInfo> > adummy4;
    //
    edm::Wrapper< std::map<unsigned int, std::vector<SiStripClusterInfo> > > adummy5;
  }
}

#endif // SISTRIPCLUSTER_CLASSES_H
