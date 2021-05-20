#ifndef __L1Trigger_VertexFinder_RecoVertex_h__
#define __L1Trigger_VertexFinder_RecoVertex_h__

#include "DataFormats/L1Trigger/interface/Vertex.h"
#include "L1Trigger/VertexFinder/interface/L1TrackTruthMatched.h"
#include "L1Trigger/VertexFinder/interface/TP.h"

#include <set>
#include <vector>

namespace l1tVertexFinder {

  template <typename T = L1Track>
  class RecoVertex {
  public:
    /// Basic constructor
    RecoVertex(const double z0 = -999.);
    /// Conversion from RecoVertex<L1Track> RecoVertex<T>
    RecoVertex(RecoVertex<L1Track>& vertex,
               std::map<const edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_>>, const T*> trackAssociationMap);
    /// Conversion from l1t::Vertex to l1tVertexFinder::RecoVertex
    RecoVertex(const l1t::Vertex& vertex,
               std::map<const edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_>>, const T*> trackAssociationMap);
    /// Basic destructor
    ~RecoVertex() {}

    /// Operators
    RecoVertex& operator+=(const RecoVertex& rhs) {
      this->tracks_.insert(std::end(this->tracks_), std::begin(rhs.tracks()), std::end(rhs.tracks()));
      return *this;
    }

    /// Clear track vector
    void clear() { tracks_.clear(); }
    /// Compute vertex parameters
    void computeParameters(unsigned int weightedmean = false, double highPtThreshold = 50., int highPtBehavior = -1);
    /// Contain high-pT track?
    bool hasHighPt() const { return highPt_; }
    /// highest track pT in the vertex
    double highestPt() const { return highestPt_; }
    /// Assign fitted track to this vertex
    void insert(const T* fitTrack) { tracks_.push_back(fitTrack); }
    /// Set primary vertex tag
    void isPrimary(bool is) { pv_ = is; }
    /// Number of high-pT tracks (pT > 10 GeV)
    unsigned int numHighPtTracks() const { return numHighPtTracks_; }
    /// Number of tracks originating from this vertex
    unsigned int numTracks() const { return tracks_.size(); }
    /// Number of true particles assigned to this vertex
    unsigned int numTrueTracks() const { return trueTracks_.size(); }
    /// True if primary vertex
    bool primaryVertex() const { return pv_; }
    /// Sum of fitted tracks transverse momentum [GeV]
    double pT() const { return pT_; }
    /// Tracking Particles in vertex
    const std::vector<const T*>& tracks() const { return tracks_; }
    /// Set z0 position [cm]
    void setZ(double z) { z0_ = z; }
    /// Vertex z0 position [cm]
    double z0() const { return z0_; }
    /// Vertex z0 width [cm]
    double z0width() const { return z0width_; }

  private:
    double z0_;
    double z0width_;
    double pT_;
    double highestPt_;
    std::vector<const T*> tracks_;
    std::set<const TP*> trueTracks_;
    bool pv_;
    bool highPt_;
    unsigned int numHighPtTracks_;
  };

  using RecoVertexWithTP = RecoVertex<L1TrackTruthMatched>;

  template <typename T>
  RecoVertex<T>::RecoVertex(const double z0) : z0_(z0) {
    z0width_ = -999.;
    pT_ = -999.;
    highestPt_ = -999.;
    pv_ = false;
    highPt_ = false;
    numHighPtTracks_ = 0;
    clear();
  }

  template <typename T>
  RecoVertex<T>::RecoVertex(RecoVertex<L1Track>& vertex,
                            std::map<const edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_>>, const T*> trackAssociationMap) {
    z0_ = -999.;
    z0width_ = -999.;
    pT_ = -999.;
    highestPt_ = -999.;
    pv_ = false;
    highPt_ = false;
    numHighPtTracks_ = 0;
    clear();

    // loop over base fitted tracks in reco vertex and find the corresponding TP
    // track using the TTTrack - L1TrackTruthMatched map from above
    for (const auto& trackIt : vertex.tracks()) {
      // using insert ensures that true tracks are also stored in vertex object
      insert(trackAssociationMap[trackIt->getTTTrackPtr()]);
    }
  }

  template <typename T>
  RecoVertex<T>::RecoVertex(const l1t::Vertex& vertex,
                            std::map<const edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_>>, const T*> trackAssociationMap) {
    z0_ = -999.;
    z0width_ = -999.;
    pT_ = -999.;
    highestPt_ = -999.;
    pv_ = false;
    highPt_ = false;
    numHighPtTracks_ = 0;
    clear();

    // populate vertex with tracks and TP track using the
    // TTTrack - L1TrackTruthMatched map from above
    for (const auto& track : vertex.tracks()) {
      // using insert ensures that true tracks are also stored in vertex object
      insert(trackAssociationMap.at(track));
    }
  }

  template <typename T>
  void RecoVertex<T>::computeParameters(unsigned int weightedmean, double highPtThreshold, int highPtBehavior) {
    pT_ = 0.;
    z0_ = 0.;
    highPt_ = false;
    highestPt_ = 0.;
    numHighPtTracks_ = 0;

    float SumZ = 0.;
    float z0square = 0.;
    float trackPt = 0.;

    for (const T* track : tracks_) {
      trackPt = track->pt();
      if (trackPt > highPtThreshold) {
        highPt_ = true;
        numHighPtTracks_++;
        highestPt_ = (trackPt > highestPt_) ? trackPt : highestPt_;
        if (highPtBehavior == 0)
          continue;  // ignore this track
        else if (highPtBehavior == 1)
          trackPt = highPtThreshold;  // saturate
      }

      pT_ += std::pow(trackPt, weightedmean);
      SumZ += track->z0() * std::pow(trackPt, weightedmean);
      z0square += track->z0() * track->z0();
    }

    z0_ = SumZ / ((weightedmean > 0) ? pT_ : tracks_.size());
    z0square /= tracks_.size();
    z0width_ = sqrt(std::abs(z0_ * z0_ - z0square));
  }

  // Template specializations
  template <>
  void RecoVertexWithTP::clear();

  template <>
  void RecoVertexWithTP::insert(const L1TrackTruthMatched* fitTrack);

}  // namespace l1tVertexFinder

#endif
