#ifndef DataFormats_L1TCalorimeterPhase2_CaloPFCluster_h
#define DataFormats_L1TCalorimeterPhase2_CaloPFCluster_h

#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace l1tp2 {

  class CaloPFCluster : public l1t::L1Candidate {
  public:
    CaloPFCluster()
        : l1t::L1Candidate(),
          clusterEt_(0.),
          clusterIEta_(-99),
          clusterIPhi_(-99),
          clusterEta_(-99.),
          clusterPhi_(-99.){};

    CaloPFCluster(const PolarLorentzVector& p4,
                  float clusterEt,
                  int clusterIEta,
                  int clusterIPhi,
                  float clusterEta,
                  float clusterPhi)
        : l1t::L1Candidate(p4),
          clusterEt_(clusterEt),
          clusterIEta_(clusterIEta),
          clusterIPhi_(clusterIPhi),
          clusterEta_(clusterEta),
          clusterPhi_(clusterPhi){};

    inline float clusterEt() const { return clusterEt_; };
    inline int clusterIEta() const { return clusterIEta_; };
    inline int clusterIPhi() const { return clusterIPhi_; };
    inline float clusterEta() const { return clusterEta_; };
    inline float clusterPhi() const { return clusterPhi_; };
    void setClusterEt(float clusterEtIn) { clusterEt_ = clusterEtIn; };
    void setClusterIEta(int clusterIEtaIn) { clusterIEta_ = clusterIEtaIn; };
    void setClusterIPhi(int clusterIPhiIn) { clusterIPhi_ = clusterIPhiIn; };
    void setClusterEta(float clusterEtaIn) { clusterEta_ = clusterEtaIn; };
    void setClusterPhi(float clusterPhiIn) { clusterPhi_ = clusterPhiIn; };

  private:
    // ET
    float clusterEt_;
    // GCT ieta
    int clusterIEta_;
    // GCT iphi
    int clusterIPhi_;
    // Tower (real) eta
    float clusterEta_;
    // Tower (real) phi
    float clusterPhi_;
  };

  // Concrete collection of output objects (with extra tuning information)
  typedef std::vector<l1tp2::CaloPFCluster> CaloPFClusterCollection;
}  // namespace l1tp2
#endif
