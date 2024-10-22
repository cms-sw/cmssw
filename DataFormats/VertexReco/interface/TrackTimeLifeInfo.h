#ifndef DataFormats_VertexReco_TrackTimeLifeInfo_h
#define DataFormats_VertexReco_TrackTimeLifeInfo_h

/**
  \class    TrackTimeLifeInfo
  \brief    Structure to hold time-life information

  \author   Michal Bluj, NCBJ, Warsaw
*/

#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

class TrackTimeLifeInfo {
public:
  TrackTimeLifeInfo();
  ~TrackTimeLifeInfo() {}

  // Secondary vertex
  void setSV(reco::Vertex sv) {
    sv_ = sv;
    hasSV_ = true;
  }
  const reco::Vertex& sv() const { return sv_; }
  bool hasSV() const { return hasSV_; }
  void setFlightVector(GlobalVector flight_vec, GlobalError flight_cov) {
    flight_vec_ = flight_vec;
    flight_cov_ = flight_cov;
  }
  // Flight-path
  const GlobalVector& flightVector() const { return flight_vec_; }
  const GlobalError& flightCovariance() const { return flight_cov_; }
  void setFlightLength(Measurement1D flightLength) { flightLength_ = flightLength; }
  const Measurement1D& flightLength() const { return flightLength_; }
  // Point of closest approach
  void setPCA(GlobalPoint pca, GlobalError pca_cov) {
    pca_ = pca;
    pca_cov_ = pca_cov;
  }
  const GlobalPoint& pca() const { return pca_; }
  const GlobalError& pcaCovariance() const { return pca_cov_; }
  // Impact parameter
  void setIP(GlobalVector ip_vec, GlobalError ip_cov) {
    ip_vec_ = ip_vec;
    ip_cov_ = ip_cov;
  }
  const GlobalVector& ipVector() const { return ip_vec_; }
  const GlobalError& ipCovariance() const { return ip_cov_; }
  void setIPLength(Measurement1D ipLength) { ipLength_ = ipLength; }
  const Measurement1D& ipLength() const { return ipLength_; }
  // Track
  void setTrack(const reco::Track* track) {
    if (track != nullptr) {
      track_ = *track;
      hasTrack_ = true;
    } else {
      track_ = reco::Track();
      hasTrack_ = false;
    }
  }
  const reco::Track* track() const { return &track_; }
  bool hasTrack() const { return hasTrack_; }
  void setBField_z(float bField_z) { bField_z_ = bField_z; }
  float bField_z() const { return bField_z_; }

private:
  bool hasSV_, hasTrack_;
  reco::Vertex sv_;
  GlobalVector flight_vec_, ip_vec_;
  GlobalPoint pca_;
  GlobalError flight_cov_, pca_cov_, ip_cov_;
  Measurement1D flightLength_, ipLength_;
  reco::Track track_;
  float bField_z_;
};

#endif
