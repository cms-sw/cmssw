#ifndef DataFormats_L1HGCal_ClusterShapes_H
#define DataFormats_L1HGCal_ClusterShapes_H
#include <cmath>

namespace l1t {
  //  this class is design to contain and compute
  //  efficiently the cluster shapes
  //  running only once on the cluster members.
  class ClusterShapes {
  private:
    float sum_e_ = 0.0;
    float sum_e2_ = 0.0;
    float sum_logE_ = 0.0;
    int n_ = 0.0;

    float emax_ = 0.0;

    float sum_w_ = 0.0;  // just here for clarity
    float sum_eta_ = 0.0;
    float sum_r_ = 0.0;
    // i will discriminate using the rms in -pi,pi or in 0,pi
    float sum_phi_0_ = 0.0;  // computed in -pi,pi
    float sum_phi_1_ = 0.0;  // computed in 0, 2pi

    float sum_eta2_ = 0.0;
    float sum_r2_ = 0.0;
    float sum_phi2_0_ = 0.0;  //computed in -pi,pi
    float sum_phi2_1_ = 0.0;  //computed in 0,2pi

    // off diagonal element of the tensor
    float sum_eta_r_ = 0.0;
    float sum_r_phi_0_ = 0.0;
    float sum_r_phi_1_ = 0.0;
    float sum_eta_phi_0_ = 0.0;
    float sum_eta_phi_1_ = 0.0;

    // caching of informations
    mutable bool isPhi0_ = true;
    mutable bool modified_ = false;  // check wheneever i need

  public:
    ClusterShapes() {}
    ClusterShapes(float e, float eta, float phi, float r) { Init(e, eta, phi, r); }
    ~ClusterShapes() {}
    ClusterShapes(const ClusterShapes &x) = default;
    //init an empty cluster
    void Init(float e, float eta, float phi, float r = 0.);
    inline void Add(float e, float eta, float phi, float r = 0.0) { (*this) += ClusterShapes(e, eta, phi, r); }

    // ---
    float SigmaEtaEta() const;
    float SigmaPhiPhi() const;
    float SigmaRR() const;
    // ----
    float Phi() const;
    float R() const;
    float Eta() const;
    inline int N() const { return n_; }
    // --
    float SigmaEtaR() const;
    float SigmaEtaPhi() const;
    float SigmaRPhi() const;
    // --
    float LogEoverE() const { return sum_logE_ / sum_e_; }
    float eD() const { return std::sqrt(sum_e2_) / sum_e_; }

    ClusterShapes operator+(const ClusterShapes &);
    void operator+=(const ClusterShapes &);
    ClusterShapes &operator=(const ClusterShapes &) = default;
  };

};  // namespace l1t

#endif
