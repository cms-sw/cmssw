#ifndef DataFormats_L1TCalorimeterPhase2_L1CaloJets_h
#define DataFormats_L1TCalorimeterPhase2_L1CaloJets_h

#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace l1tp2 {

  class L1CaloJet : public l1t::L1Candidate {
  public:
    L1CaloJet() : l1t::L1Candidate(), calibratedPt_(0.), hovere_(0.), iso_(0.), pUcorrPt_(0.){};

    L1CaloJet(const PolarLorentzVector& p4, float calibratedPt, float hovere, float iso, float PUcorrPt = 0.)
        : l1t::L1Candidate(p4), calibratedPt_(calibratedPt), hovere_(hovere), iso_(iso), pUcorrPt_(PUcorrPt){};

    inline float calibratedPt() const { return calibratedPt_; };
    inline float hovere() const { return hovere_; };
    inline float isolation() const { return iso_; };
    inline float pUcorrPt() const { return pUcorrPt_; };
    std::vector<std::vector<float>>& associated_l1EGs() { return associated_l1EGs_; };

    void setExperimentalParams(const std::map<std::string, float>& params) { experimentalParams_ = params; };
    void setAssociated_l1EGs(const std::vector<std::vector<float>> l1EGs) { associated_l1EGs_ = l1EGs; };

    const std::map<std::string, float>& experimentalParams() const { return experimentalParams_; };

    inline float experimentalParam(std::string const& name) const {
      if (experimentalParams_.count(name)) {
        return experimentalParams_.at(name);
      } else {
        edm::LogError("L1CaloJet") << "Error: no mapping for ExperimentalParam: " << name << std::endl;
        return -99.;
      }
    };

  private:
    // pT calibrated to get
    float calibratedPt_;
    // HCal energy in region behind cluster (for size, look in producer) / ECal energy in cluster
    float hovere_;
    // ECal isolation (for outer window size, again look in producer)
    float iso_;
    // Pileup-corrected energy deposit, not studied carefully yet, don't use
    float pUcorrPt_;
    // For investigating novel algorithm parameters
    std::map<std::string, float> experimentalParams_;
    // For decay mode related checks with CaloTaus
    std::vector<std::vector<float>> associated_l1EGs_;
  };

  // Concrete collection of output objects (with extra tuning information)
  typedef std::vector<l1tp2::L1CaloJet> L1CaloJetsCollection;
}  // namespace l1tp2
#endif
