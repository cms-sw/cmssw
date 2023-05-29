// Author: Felice Pantaleo - felice.pantaleo@cern.ch
// Date: 09/2018

#ifndef DataFormats_HGCalReco_Trackster_h
#define DataFormats_HGCalReco_Trackster_h

#include <array>
#include <vector>
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Math/interface/Vector3D.h"

#include <Eigen/Core>

// A Trackster is a Direct Acyclic Graph created when
// pattern recognition algorithms connect hits or
// layer clusters together in a 3D object.

namespace ticl {
  class Trackster {
  public:
    typedef math::XYZVector Vector;

    enum IterationIndex { TRKEM = 0, EM, TRKHAD, HAD, MIP, SIM, SIM_CP };

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

    enum class PCAOrdering { ascending = 0, descending };

    Trackster()
        : iterationIndex_(0),
          seedIndex_(-1),
          time_(0.f),
          timeError_(-1.f),
          regressed_energy_(0.f),
          raw_energy_(0.f),
          raw_em_energy_(0.f),
          raw_pt_(0.f),
          raw_em_pt_(0.f),
          barycenter_({0., 0., 0.}),
          eigenvalues_{{0.f, 0.f, 0.f}},
          sigmas_{{0.f, 0.f, 0.f}},
          sigmasPCA_{{0.f, 0.f, 0.f}} {
      zeroProbabilities();
    }

    inline void setIteration(const Trackster::IterationIndex i) { iterationIndex_ = i; }
    std::vector<unsigned int> &vertices() { return vertices_; }
    std::vector<float> &vertex_multiplicity() { return vertex_multiplicity_; }
    std::vector<std::array<unsigned int, 2> > &edges() { return edges_; }
    inline void setSeed(edm::ProductID pid, int index) {
      seedID_ = pid;
      seedIndex_ = index;
    }
    inline void setTimeAndError(float t, float tError) {
      time_ = t;
      timeError_ = tError;
    }
    inline void setRegressedEnergy(float value) { regressed_energy_ = value; }
    inline void setRawEnergy(float value) { raw_energy_ = value; }
    inline void addToRawEnergy(float value) { raw_energy_ += value; }
    inline void setRawEmEnergy(float value) { raw_em_energy_ = value; }
    inline void addToRawEmEnergy(float value) { raw_em_energy_ += value; }
    inline void setRawPt(float value) { raw_pt_ = value; }
    inline void setRawEmPt(float value) { raw_em_pt_ = value; }
    inline void setBarycenter(Vector value) { barycenter_ = value; }
    inline void fillPCAVariables(Eigen::Vector3d &eigenvalues,
                                 Eigen::Matrix3d &eigenvectors,
                                 Eigen::Vector3d &sigmas,
                                 Eigen::Vector3d &sigmasEigen,
                                 size_t pcadimension,
                                 PCAOrdering order) {
      int original_index = 0;
      for (size_t i = 0; i < pcadimension; ++i) {
        sigmas_[i] = std::sqrt(sigmas[i]);
        // Reverse the order, since Eigen gives back the eigevalues in
        // **increasing order** while we store them in **descreasing order**.
        original_index = i;
        if (order == PCAOrdering::ascending)
          original_index = pcadimension - i - 1;
        eigenvalues_[i] = (float)eigenvalues[original_index];
        eigenvectors_[i] = ticl::Trackster::Vector(
            eigenvectors(0, original_index), eigenvectors(1, original_index), eigenvectors(2, original_index));
        sigmasPCA_[i] = std::sqrt(sigmasEigen[original_index]);
      }
      original_index = 0;
      if (order == PCAOrdering::ascending)
        original_index = pcadimension - 1;
      if (eigenvectors_[0].z() * barycenter_.z() < 0.0) {
        eigenvectors_[0] = -ticl::Trackster::Vector(
            eigenvectors(0, original_index), eigenvectors(1, original_index), eigenvectors(2, original_index));
      }

      // Now also update the pt part of the Trackster, using the PCA as direction
      raw_pt_ = std::sqrt((eigenvectors_[0].Unit() * raw_energy_).perp2());
      raw_em_pt_ = std::sqrt((eigenvectors_[0].Unit() * raw_em_energy_).perp2());
    }
    void zeroProbabilities() {
      for (auto &p : id_probabilities_) {
        p = 0.f;
      }
    }
    inline void setProbabilities(float *probs) {
      for (float &p : id_probabilities_) {
        p = *(probs++);
      }
    }
    inline void setIdProbability(ParticleType type, float value) { id_probabilities_[int(type)] = value; }

    inline const Trackster::IterationIndex ticlIteration() const { return (IterationIndex)iterationIndex_; }
    inline const std::vector<unsigned int> &vertices() const { return vertices_; }
    inline const unsigned int vertices(int index) const { return vertices_[index]; }
    inline const std::vector<float> &vertex_multiplicity() const { return vertex_multiplicity_; }
    inline const float vertex_multiplicity(int index) const { return vertex_multiplicity_[index]; }
    inline const std::vector<std::array<unsigned int, 2> > &edges() const { return edges_; }
    inline const edm::ProductID &seedID() const { return seedID_; }
    inline const int seedIndex() const { return seedIndex_; }
    inline const float time() const { return time_; }
    inline const float timeError() const { return timeError_; }
    inline const float regressed_energy() const { return regressed_energy_; }
    inline const float raw_energy() const { return raw_energy_; }
    inline const float raw_em_energy() const { return raw_em_energy_; }
    inline const float raw_pt() const { return raw_pt_; }
    inline const float raw_em_pt() const { return raw_em_pt_; }
    inline const Vector &barycenter() const { return barycenter_; }
    inline const std::array<float, 3> &eigenvalues() const { return eigenvalues_; }
    inline const std::array<Vector, 3> &eigenvectors() const { return eigenvectors_; }
    inline const Vector &eigenvectors(int index) const { return eigenvectors_[index]; }
    inline const std::array<float, 3> &sigmas() const { return sigmas_; }
    inline const std::array<float, 3> &sigmasPCA() const { return sigmasPCA_; }
    inline const std::array<float, 8> &id_probabilities() const { return id_probabilities_; }
    inline const float id_probabilities(int index) const { return id_probabilities_[index]; }

    // convenience method to return the ID probability for a certain particle type
    inline float id_probability(ParticleType type) const {
      // probabilities are stored in the same order as defined in the ParticleType enum
      return id_probabilities_[(int)type];
    }

  private:
    // TICL iteration producing the trackster
    uint8_t iterationIndex_;

    // The vertices of the DAG are the indices of the
    // 2d objects in the global collection
    std::vector<unsigned int> vertices_;
    std::vector<float> vertex_multiplicity_;

    // The edges connect two vertices together in a directed doublet
    // ATTENTION: order matters!
    // A doublet generator should create edges in which:
    // the first element is on the inner layer and
    // the outer element is on the outer layer.
    std::vector<std::array<unsigned int, 2> > edges_;

    // Product ID of the seeding collection used to create the Trackster.
    // For GlobalSeeding the ProductID is set to 0. For track-based seeding
    // this is the ProductID of the track-collection used to create the
    // seeding-regions.
    edm::ProductID seedID_;

    // For Global Seeding the index is fixed to one. For track-based seeding,
    // the index is the index of the track originating the seeding region that
    // created the trackster. For track-based seeding the pointer to the track
    // can be cooked using the previous ProductID and this index.
    int seedIndex_;

    // We also need the pointer to the original seeding region ??
    // something like:
    // int seedingRegionIdx;

    // -99, -1 if not available. ns units otherwise
    float time_;
    float timeError_;

    // regressed energy
    float regressed_energy_;
    float raw_energy_;
    float raw_em_energy_;
    float raw_pt_;
    float raw_em_pt_;

    // PCA Variables
    Vector barycenter_;
    std::array<float, 3> eigenvalues_;
    std::array<Vector, 3> eigenvectors_;
    std::array<float, 3> sigmas_;
    std::array<float, 3> sigmasPCA_;

    // trackster ID probabilities
    std::array<float, 8> id_probabilities_;
  };

  typedef std::vector<Trackster> TracksterCollection;
}  // namespace ticl
#endif
