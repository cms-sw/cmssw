#ifndef DataFormats_CLUSTERFP420_CLASSES_H
#define DataFormats_CLUSTERFP420_CLASSES_H

#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include <vector>
#include <string>

///////////////////////////////////////////////////////////////////////////////

#include "DataFormats/FP420Cluster/interface/ClusterFP420.h"
#include "DataFormats/FP420Cluster/interface/ClusterCollectionFP420.h"
namespace {
  struct dictionary {
    edm::Wrapper<ClusterFP420 > zc0;
    edm::Wrapper<std::vector<ClusterFP420> > zc1;
    edm::Wrapper< edm::DetSet<ClusterFP420> > zc2;
    edm::Wrapper< std::vector<edm::DetSet<ClusterFP420> > > zc3;
    edm::Wrapper< edm::DetSetVector<ClusterFP420> > zc4;

    edm::Wrapper<ClusterCollectionFP420> ccollection;
  };
}

///////////////////////////////////////////////////////////////////////////////

#include "DataFormats/FP420Cluster/interface/TrackFP420.h"
#include "DataFormats/FP420Cluster/interface/TrackCollectionFP420.h"
namespace {
  struct dictionary2 {
    edm::Wrapper<TrackFP420 > zt0;
    edm::Wrapper<std::vector<TrackFP420> > zt1;
    edm::Wrapper< edm::DetSet<TrackFP420> > zt2;
    edm::Wrapper< std::vector<edm::DetSet<TrackFP420> > > zt3;
    edm::Wrapper< edm::DetSetVector<TrackFP420> > zt4;

    edm::Wrapper<TrackCollectionFP420> tcollection;
  };
}

///////////////////////////////////////////////////////////////////////////////

#include "DataFormats/FP420Cluster/interface/RecoFP420.h"
#include "DataFormats/FP420Cluster/interface/RecoCollectionFP420.h"
namespace {
  struct dictionary3 {
    edm::Wrapper<RecoFP420 > zr0;
    edm::Wrapper<std::vector<RecoFP420> > zr1;
    edm::Wrapper< edm::DetSet<RecoFP420> > zr2;
    edm::Wrapper< std::vector<edm::DetSet<RecoFP420> > > zr3;
    edm::Wrapper< edm::DetSetVector<RecoFP420> > zr4;

    edm::Wrapper<RecoCollectionFP420> rcollection;
  };
}

///////////////////////////////////////////////////////////////////////////////

#endif // ClusterFP420_CLASSES_H


