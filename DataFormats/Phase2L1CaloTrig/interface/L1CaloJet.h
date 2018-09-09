#ifndef L1CaloJets_h
#define L1CaloJets_h

#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include "DataFormats/L1Trigger/interface/L1Candidate.h"

namespace l1slhc
{

  class L1CaloJet : public l1t::L1Candidate {
    public:
      L1CaloJet() : l1t::L1Candidate(), calibratedPt_(0.), hovere_(0.), iso_(0.), PUcorrPt_(0.) {};

      L1CaloJet(const PolarLorentzVector& p4, float calibratedPt, float hovere, float iso, 
            float PUcorrPt = 0. ) :
                    l1t::L1Candidate(p4), calibratedPt_(calibratedPt), hovere_(hovere), iso_(iso),
                    PUcorrPt_(PUcorrPt) {};

      virtual ~L1CaloJet() {};
      inline float calibratedPt() const { return calibratedPt_; };
      inline float hovere() const { return hovere_; };
      inline float isolation() const { return iso_; };
      inline float PUcorrPt() const { return PUcorrPt_; };
      void SetExperimentalParams(const std::map<std::string, float> &params) { experimentalParams_ = params; };
      const std::map<std::string, float> GetExperimentalParams() const { return experimentalParams_; };
      inline float GetExperimentalParam(std::string name) const {
         try { return experimentalParams_.at(name); }
         catch (...) { 
            std::cout << "Error: no mapping for ExperimentalParam: " << name << std::endl;
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
      float PUcorrPt_;
      // For investigating novel algorithm parameters
      std::map<std::string, float> experimentalParams_;
  };
  
  
  // Concrete collection of output objects (with extra tuning information)
  typedef std::vector<l1slhc::L1CaloJet> L1CaloJetsCollection;
}
#endif

