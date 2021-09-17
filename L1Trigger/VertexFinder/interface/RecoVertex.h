#ifndef __L1Trigger_VertexFinder_RecoVertex_h__
#define __L1Trigger_VertexFinder_RecoVertex_h__

#include "DataFormats/L1Trigger/interface/Vertex.h"
#include "L1Trigger/VertexFinder/interface/L1TrackTruthMatched.h"
#include "L1Trigger/VertexFinder/interface/TP.h"

#include <numeric>
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
    //void computeParameters(unsigned int weightedmean = false, double highPtThreshold = 50., int highPtBehavior = -1);
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
    double pt() const { return pT_; }
    /// Tracks in the vertex
    const std::vector<const T*>& tracks() const { return tracks_; }
    /// Tracking particles asociated to the vertex
    const std::set<const TP*>& trueTracks() const { return trueTracks_; }
    /// set the pT [GeV] of the vertex
    void setPt(double pt) { pT_ = pt; }
    /// Set z0 position [cm]
    void setZ0(double z) { z0_ = z; }
    /// Set the vertex parameters
    void setParameters(double pt,
                       double z0,
                       double width = -999.,
                       bool highPt = false,
                       unsigned int nHighPt = -999,
                       double highestPt = -999.,
                       bool pv = false);
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
    z0_ = vertex.z0();
    z0width_ = vertex.z0width();
    pT_ = vertex.pt();
    highestPt_ = vertex.highestPt();
    pv_ = vertex.primaryVertex();
    highPt_ = vertex.hasHighPt();
    numHighPtTracks_ = vertex.numHighPtTracks();
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
    z0_ = vertex.z0();
    z0width_ = -999.;
    pT_ = vertex.pt();
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
  void RecoVertex<T>::setParameters(
      double pt, double z0, double width, bool highPt, unsigned int nHighPt, double highestPt, bool pv) {
    pT_ = pt;
    z0_ = z0;
    z0width_ = width;
    highPt_ = highPt;
    numHighPtTracks_ = nHighPt;
    highestPt_ = highestPt;
    pv_ = pv;
  }

  // Template specializations
  template <>
  RecoVertexWithTP& RecoVertexWithTP::operator+=(const RecoVertexWithTP& rhs);

  template <>
  void RecoVertexWithTP::clear();

  template <>
  void RecoVertexWithTP::insert(const L1TrackTruthMatched* fitTrack);

}  // namespace l1tVertexFinder

#endif
