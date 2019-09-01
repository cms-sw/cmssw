// Author: Felice Pantaleo - felice.pantaleo@cern.ch
// Date: 09/2018

#ifndef DataFormats_HGCalReco_Trackster_h
#define DataFormats_HGCalReco_Trackster_h

#include <array>
#include <vector>
#include "DataFormats/Provenance/interface/ProductID.h"

// A Trackster is a Direct Acyclic Graph created when
// pattern recognition algorithms connect hits or
// layer clusters together in a 3D object.

namespace ticl {
  struct Trackster {
    // The vertices of the DAG are the indices of the
    // 2d objects in the global collection
    std::vector<unsigned int> vertices;
    std::vector<uint8_t> vertex_multiplicity;

    // The edges connect two vertices together in a directed doublet
    // ATTENTION: order matters!
    // A doublet generator should create edges in which:
    // the first element is on the inner layer and
    // the outer element is on the outer layer.
    std::vector<std::array<unsigned int, 2> > edges;

    edm::ProductID seedID;
    int seedIndex;
  };
}  // namespace ticl
#endif
