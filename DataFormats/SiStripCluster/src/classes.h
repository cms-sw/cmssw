#ifndef SISTRIPCLUSTER_CLASSES_H
#define SISTRIPCLUSTER_CLASSES_H

#include "DataFormats/SiStripCluster/interface/SiStripClusterCollection.h"
#include "DataFormats/SiStripCluster/interface/SiStrip1DLocalMeasurementCollection.h"
#include "FWCore/EDProduct/interface/Wrapper.h"
#include <vector>

namespace {
  namespace {
    edm::Wrapper<SiStripClusterCollection> siStripClusterCollectionWrapper;
    edm::Wrapper<SiStrip1DLocalMeasurementCollection> siStrip1DLocalMeasurementCollectionWrapper;
  }
}

#endif // SISTRIPCLUSTER_CLASSES_H
