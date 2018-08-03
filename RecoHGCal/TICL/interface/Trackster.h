// Author: Felice Pantaleo - felice.pantaleo@cern.ch
// Date: 09/2018
// Copyright CERN

#ifndef __RecoHGCal_TICL_Trackster_H__
#define __RecoHGCal_TICL_Trackster_H__

#include <vector>
#include <array>

// A Trackster is a Direct Acyclic Graph created when
// pattern recognition algorithms connect hits or
// layer clusters together in a 3D object.

struct Trackster {
    // // The type of the objects that the edges connect
    // enum class VertexType = {
    //     HGCALLAYERCLUSTER,
    //     HGCRECHIT
    // };


    // The vertices of the DAG are the indices of the
    // 2d objects in the global collection
    std::vector<unsigned int> vertices;

    // The edges connect two vertices together in a directed doublet
    // ATTENTION: order matters!
    // A doublet generator should create edges in which:
    // the first element is on the inner layer and
    // the outer element is on the outer layer.
    std::vector<std::array<unsigned int, 2> > edges;
};

#endif
