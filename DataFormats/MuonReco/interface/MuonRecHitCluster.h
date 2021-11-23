#ifndef DataFormats_MuonReco_MuonRecHitCluster_h
#define DataFormats_MuonReco_MuonRecHitCluster_h

#include <vector>
#include "DataFormats/Math/interface/Vector3D.h"

namespace reco {

  class MuonRecHitCluster {
  public:
    //default constructor
    MuonRecHitCluster() = default;

    MuonRecHitCluster(const math::RhoEtaPhiVectorF position,
                      const int size,
                      const int nStation,
                      const float avgStation,
                      const float time,
                      const float timeSpread,
                      const int nME11,
                      const int nME12,
                      const int nME41,
                      const int nME42,
                      const int nMB1,
                      const int nMB2);

    //
    ~MuonRecHitCluster() = default;

    float eta() const { return position_.Eta(); }
    float phi() const { return position_.Phi(); }
    float x() const { return position_.X(); }
    float y() const { return position_.Y(); }
    float z() const { return position_.Z(); }
    float r() const { return position_.Rho(); }
    int size() const { return size_; }
    int nStation() const { return nStation_; }
    float avgStation() const { return avgStation_; }
    int nMB1() const { return nMB1_; }
    int nMB2() const { return nMB2_; }
    int nME11() const { return nME11_; }
    int nME12() const { return nME12_; }
    int nME41() const { return nME41_; }
    int nME42() const { return nME42_; }
    float time() const { return time_; }
    float timeSpread() const { return timeSpread_; }

  private:
    math::RhoEtaPhiVectorF position_;
    int size_;
    int nStation_;
    float avgStation_;
    float time_;
    float timeSpread_;
    int nME11_;
    int nME12_;
    int nME41_;
    int nME42_;
    int nMB1_;
    int nMB2_;
  };

  typedef std::vector<MuonRecHitCluster> MuonRecHitClusterCollection;
}  // namespace reco
#endif
