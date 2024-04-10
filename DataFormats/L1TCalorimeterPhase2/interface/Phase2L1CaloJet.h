#ifndef DataFormats_L1TCalorimeterPhase2_Phase2L1CaloJet_h
#define DataFormats_L1TCalorimeterPhase2_Phase2L1CaloJet_h

#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace l1tp2 {

  class Phase2L1CaloJet : public l1t::L1Candidate {
  public:
    Phase2L1CaloJet()
        : l1t::L1Candidate(),
          jetEt_(0.),
          tauEt_(0.),
          jetIEta_(-99),
          jetIPhi_(-99),
          jetEta_(-99.),
          jetPhi_(-99.),
          towerEt_(0.),
          towerIEta_(-99),
          towerIPhi_(-99),
          towerEta_(-99.),
          towerPhi_(-99.){};

    Phase2L1CaloJet(const PolarLorentzVector& p4,
                    float jetEt,
                    float tauEt,
                    int jetIEta,
                    int jetIPhi,
                    float jetEta,
                    float jetPhi,
                    float towerEt,
                    int towerIEta,
                    int towerIPhi,
                    float towerEta,
                    float towerPhi)
        : l1t::L1Candidate(p4),
          jetEt_(jetEt),
          tauEt_(tauEt),
          jetIEta_(jetIEta),
          jetIPhi_(jetIPhi),
          jetEta_(jetEta),
          jetPhi_(jetPhi),
          towerEt_(towerEt),
          towerIEta_(towerIEta),
          towerIPhi_(towerIPhi),
          towerEta_(towerEta),
          towerPhi_(towerPhi){};

    inline float jetEt() const { return jetEt_; };
    inline float tauEt() const { return tauEt_; };
    inline int jetIEta() const { return jetIEta_; };
    inline int jetIPhi() const { return jetIPhi_; };
    inline float jetEta() const { return jetEta_; };
    inline float jetPhi() const { return jetPhi_; };
    inline float towerEt() const { return towerEt_; };
    inline int towerIEta() const { return towerIEta_; };
    inline int towerIPhi() const { return towerIPhi_; };
    inline float towerEta() const { return towerEta_; };
    inline float towerPhi() const { return towerPhi_; };

    void setJetEt(float jetEtIn) { jetEt_ = jetEtIn; };
    void setTauEt(float tauEtIn) { tauEt_ = tauEtIn; };
    void setJetIEta(int jetIEtaIn) { jetIEta_ = jetIEtaIn; };
    void setJetIPhi(int jetIPhiIn) { jetIPhi_ = jetIPhiIn; };
    void setJetEta(float jetEtaIn) { jetEta_ = jetEtaIn; };
    void setJetPhi(float jetPhiIn) { jetPhi_ = jetPhiIn; };
    void setTowerEt(float towerEtIn) { towerEt_ = towerEtIn; };
    void setTowerIEta(int towerIEtaIn) { towerIEta_ = towerIEtaIn; };
    void setTowerIPhi(int towerIPhiIn) { towerIPhi_ = towerIPhiIn; };
    void setTowerEta(float towerEtaIn) { towerEta_ = towerEtaIn; };
    void setTowerPhi(float towerPhiIn) { towerPhi_ = towerPhiIn; };

  private:
    // ET
    float jetEt_;
    // Tau ET
    float tauEt_;
    // GCT ieta
    int jetIEta_;
    // GCT iphi
    int jetIPhi_;
    // Tower (real) eta
    float jetEta_;
    // Tower (real) phi
    float jetPhi_;
    float towerEt_;
    int towerIEta_;
    int towerIPhi_;
    float towerEta_;
    float towerPhi_;
  };

  // Concrete collection of output objects (with extra tuning information)
  typedef std::vector<l1tp2::Phase2L1CaloJet> Phase2L1CaloJetCollection;
}  // namespace l1tp2
#endif
