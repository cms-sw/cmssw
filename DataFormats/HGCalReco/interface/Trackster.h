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

    // Product ID of the seeding collection used to create the Trackster.
    // For GlobalSeeding the ProductID is set to 0. For track-based seeding
    // this is the ProductID of the track-collection used to create the
    // seeding-regions.
    edm::ProductID seedID;
    // For Global Seeding the index is fixed to one. For track-based seeding,
    // the index is the index of the track originating the seeding region that
    // created the trackster. For track-based seeding the pointer to the track
    // can be cooked using the previous ProductID and this index.
    int seedIndex;

    // -99, -1 if not available. ns units otherwise
    float time;
    float timeError;

    // regressed energy
    float regressed_energy;

    // types considered by the particle identification
    enum class ParticleType {
      photon = 0,
      electron,
      muon,
      neutral_pion,
      charged_hadron,
      neutral_hadron,
      ambiguous,
      unknown,
    };

    // trackster ID probabilities
    std::array<float, 8> id_probabilities;

    // convenience method to return the ID probability for a certain particle type
    inline float id_probability(ParticleType type) const {
      // probabilities are stored in the same order as defined in the ParticleType enum
      return id_probabilities[(int)type];
    }
  };
}  // namespace ticl
#endif
