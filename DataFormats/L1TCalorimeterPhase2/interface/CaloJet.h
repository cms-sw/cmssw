#ifndef DataFormats_L1TCalorimeterPhase2_CaloJets_h
#define DataFormats_L1TCalorimeterPhase2_CaloJets_h

#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include "DataFormats/L1Trigger/interface/L1Candidate.h"

namespace l1tp2 {

  class CaloJet : public l1t::L1Candidate {
  public:
    CaloJet() : l1t::L1Candidate(), calibratedPt_(0.), hovere_(0.), iso_(0.), puCorrPt_(0.){};

    CaloJet(const PolarLorentzVector& p4, float calibratedPt, float hovere, float iso, float puCorrPt = 0.)
        : l1t::L1Candidate(p4), calibratedPt_(calibratedPt), hovere_(hovere), iso_(iso), puCorrPt_(puCorrPt){};

    inline float calibratedPt() const { return calibratedPt_; };
    inline float hovere() const { return hovere_; };
    inline float isolation() const { return iso_; };
    inline float puCorrPt() const { return puCorrPt_; };
    const std::vector<std::vector<float>>& associated_l1EGs() const { return associated_l1EGs_; };

    void setExperimentalParams(const std::map<std::string, float>& params) { experimentalParams_ = params; };
    void setAssociated_l1EGs(const std::vector<std::vector<float>> l1EGs) { associated_l1EGs_ = l1EGs; };

    const std::map<std::string, float>& experimentalParams() const { return experimentalParams_; };

    inline float experimentalParam(std::string const& name) const {
      auto iter = experimentalParams_.find(name);
      if (iter != experimentalParams_.end()) {
        return iter->second;
      } else {
        warningNoMapping(name);
        return -99.;
      }
    };

  private:
    static void warningNoMapping(std::string const&);
    // pT calibrated to get
    float calibratedPt_;
    // HCal energy in region behind cluster (for size, look in producer) / ECal energy in cluster
    float hovere_;
    // ECal isolation (for outer window size, again look in producer)
    float iso_;
    // Pileup-corrected energy deposit, not studied carefully yet, don't use
    float puCorrPt_;
    // For investigating novel algorithm parameters
    std::map<std::string, float> experimentalParams_;
    // For decay mode related checks with CaloTaus
    std::vector<std::vector<float>> associated_l1EGs_;
  };

  // Concrete collection of output objects (with extra tuning information)
  typedef std::vector<l1tp2::CaloJet> CaloJetsCollection;
}  // namespace l1tp2
#endif
