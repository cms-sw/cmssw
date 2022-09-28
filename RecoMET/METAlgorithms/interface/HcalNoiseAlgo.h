#ifndef _RECOMET_METALGORITHMS_HCALNOISEALGO_H_
#define _RECOMET_METALGORITHMS_HCALNOISEALGO_H_

#include "DataFormats/METReco/interface/HcalNoiseRBX.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class CommonHcalNoiseRBXData {
public:
  CommonHcalNoiseRBXData(const reco::HcalNoiseRBX &rbx,
                         double minRecHitE,
                         double minLowHitE,
                         double minHighHitE,
                         double TS4TS5EnergyThreshold,
                         std::vector<std::pair<double, double> > const &TS4TS5UpperCut,
                         std::vector<std::pair<double, double> > const &TS4TS5LowerCut,
                         double MinRBXRechitR45E);
  ~CommonHcalNoiseRBXData() {}

  // accessors to internal variables
  inline double energy(void) const { return energy_; }
  inline double ratio(void) const { return e2ts_ / e10ts_; }
  inline double e2ts(void) const { return e2ts_; }
  inline double e10ts(void) const { return e10ts_; }
  inline bool validRatio(void) const { return e10ts_ != 0.0; }
  inline int numHPDHits(void) const { return numHPDHits_; }
  inline int numRBXHits(void) const { return numRBXHits_; }
  inline int numHPDNoOtherHits(void) const { return numHPDNoOtherHits_; }
  inline int numZeros(void) const { return numZeros_; }
  inline double minLowEHitTime(void) const { return minLowEHitTime_; }
  inline double maxLowEHitTime(void) const { return maxLowEHitTime_; }
  inline double lowEHitTimeSqrd(void) const { return lowEHitTimeSqrd_; }
  inline int numLowEHits(void) const { return numLowEHits_; }
  inline double minHighEHitTime(void) const { return minHighEHitTime_; }
  inline double maxHighEHitTime(void) const { return maxHighEHitTime_; }
  inline double highEHitTimeSqrd(void) const { return highEHitTimeSqrd_; }
  inline int numHighEHits(void) const { return numHighEHits_; }
  inline double RBXEMF(void) const { return RBXEMF_; }
  inline double HPDEMF(void) const { return HPDEMF_; }
  inline bool PassTS4TS5(void) const { return TS4TS5Decision_; }
  inline edm::RefVector<CaloTowerCollection> rbxTowers(void) const { return rbxtowers_; }
  inline int r45Count(void) const { return r45Count_; }
  inline double r45Fraction(void) const { return r45Fraction_; }
  inline double r45EnergyFraction(void) const { return r45EnergyFraction_; }

  bool CheckPassFilter(double Charge,
                       double Discriminant,
                       std::vector<std::pair<double, double> > const &Cuts,
                       int Side);

private:
  // values
  double energy_;            // RBX hadronic energy as determined by the sum of calotowers
  double e2ts_;              // pedestal subtracted charge in two peak TS for RBX
  double e10ts_;             // pedestal subtracted charge in all 10 TS for RBX
  int numHPDHits_;           // largest number of hits in an HPD in the RBX
  int numRBXHits_;           // number of hits in the RBX
  int numHPDNoOtherHits_;    // largest number of hits in an HPD when no other HPD has a hit in the RBX
  int numZeros_;             // number of ADC 0 counts in all hits in all TS in the RBX
  double minLowEHitTime_;    // minimum time found for any low energy hit in the RBX
  double maxLowEHitTime_;    // maximum time found for any low energy hit in the RBX
  double lowEHitTimeSqrd_;   // low energy hit time^2
  int numLowEHits_;          // number of low energy hits
  double minHighEHitTime_;   // minimum time found for any high energy hit in the RBX
  double maxHighEHitTime_;   // maximum time found for any high energy hit in the RBX
  double highEHitTimeSqrd_;  // high energy hit time^2
  int numHighEHits_;         // number of high energy hits
  double HPDEMF_;            // minimum electromagnetic fraction found in an HPD in the RBX
  double RBXEMF_;            // electromagnetic fraction of the RBX
  bool TS4TS5Decision_;      // if this RBX fails TS4TS5 variable or not
  edm::RefVector<CaloTowerCollection> rbxtowers_;  // calotowers associated with the RBX
  int r45Count_;                                   // Number of rechits above some threshold flagged by R45
  double r45Fraction_;                             // Fraction of rechits above some threshold flagged by R45
  double r45EnergyFraction_;                       // Energy fraction of rechits above some threshold
};

class HcalNoiseAlgo {
public:
  HcalNoiseAlgo(const edm::ParameterSet &iConfig);
  virtual ~HcalNoiseAlgo() {}

  // an rbx is "interesting/problematic" (i.e. is recorded to the event record)
  bool isProblematic(const CommonHcalNoiseRBXData &) const;

  // an rbx passes a noise filter
  bool passLooseNoiseFilter(const CommonHcalNoiseRBXData &) const;
  bool passTightNoiseFilter(const CommonHcalNoiseRBXData &) const;
  bool passHighLevelNoiseFilter(const CommonHcalNoiseRBXData &) const;

  // loose filter broken down into separate components
  bool passLooseRatio(const CommonHcalNoiseRBXData &) const;
  bool passLooseHits(const CommonHcalNoiseRBXData &) const;
  bool passLooseZeros(const CommonHcalNoiseRBXData &) const;
  bool passLooseTiming(const CommonHcalNoiseRBXData &) const;
  bool passLooseRBXRechitR45(const CommonHcalNoiseRBXData &) const;

  // tight filter broken down into separate components
  bool passTightRatio(const CommonHcalNoiseRBXData &) const;
  bool passTightHits(const CommonHcalNoiseRBXData &) const;
  bool passTightZeros(const CommonHcalNoiseRBXData &) const;
  bool passTightTiming(const CommonHcalNoiseRBXData &) const;
  bool passTightRBXRechitR45(const CommonHcalNoiseRBXData &) const;

  // an rbx passes an energy (or other) threshold to test a certain variable
  // for instance, the EMF cut might require that the RBX have 20 GeV of energy
  bool passRatioThreshold(const CommonHcalNoiseRBXData &) const;
  bool passZerosThreshold(const CommonHcalNoiseRBXData &) const;
  bool passEMFThreshold(const CommonHcalNoiseRBXData &) const;

private:
  // energy thresholds used for problematic cuts
  double pMinERatio_;  // minimum energy to apply ratio cuts
  double pMinEZeros_;  // minimum energy to apply zeros cuts
  double pMinEEMF_;    // minimum energy to apply EMF cuts

  // energy thresholds used for loose, tight and high level cuts
  double minERatio_;  // minimum energy to apply ratio cuts
  double minEZeros_;  // minimum energy to apply zeros cuts
  double minEEMF_;    // minimum energy to apply EMF cuts

  // "problematic" cuts
  // used to determine whether an RBX is stored in the EDM
  double pMinE_;                           // minimum energy
  double pMinRatio_;                       // minimum ratio
  double pMaxRatio_;                       // maximum ratio
  int pMinHPDHits_;                        // minimum # of HPD hits
  int pMinRBXHits_;                        // minimum # of RBX hits
  int pMinHPDNoOtherHits_;                 // minimum # of HPD hits with no other hits in the RBX
  int pMinZeros_;                          // minimum # of zeros
  double pMinLowEHitTime_;                 // minimum low energy hit time
  double pMaxLowEHitTime_;                 // maximum low energy hit time
  double pMinHighEHitTime_;                // minimum high energy hit time
  double pMaxHighEHitTime_;                // maximum high energy hit time
  double pMaxHPDEMF_;                      // maximum HPD EMF
  double pMaxRBXEMF_;                      // maximum RBX EMF
  int pMinRBXRechitR45Count_;              // number of R45-flagged hits
  double pMinRBXRechitR45Fraction_;        // fraction of R45-flagged hits
  double pMinRBXRechitR45EnergyFraction_;  // energy fraction of R45-flagged hits

  // "loose" cuts
  // used to determine whether an RBX fails the loose noise cuts
  double lMinRatio_;         // minimum ratio
  double lMaxRatio_;         // maximum ratio
  int lMinHPDHits_;          // minimum # of HPD hits
  int lMinRBXHits_;          // minimum # of RBX hits
  int lMinHPDNoOtherHits_;   // minimum # of HPD hits with no other hits in the RBX
  int lMinZeros_;            // minimum # of zeros
  double lMinLowEHitTime_;   // minimum low energy hit time
  double lMaxLowEHitTime_;   // maximum low energy hit time
  double lMinHighEHitTime_;  // minimum high energy hit time
  double lMaxHighEHitTime_;  // maximum high energy hit time
  std::vector<double> lMinRBXRechitR45Cuts_;

  // "tight" cuts
  // used to determine whether an RBX fails the tight noise cuts
  double tMinRatio_;         // minimum ratio
  double tMaxRatio_;         // maximum ratio
  int tMinHPDHits_;          // minimum # of HPD hits
  int tMinRBXHits_;          // minimum # of RBX hits
  int tMinHPDNoOtherHits_;   // minimum # of HPD hits with no other hits in the RBX
  int tMinZeros_;            // minimum # of zeros
  double tMinLowEHitTime_;   // minimum low energy hit time
  double tMaxLowEHitTime_;   // maximum low energy hit time
  double tMinHighEHitTime_;  // minimum high energy hit time
  double tMaxHighEHitTime_;  // maximum high energy hit time
  std::vector<double> tMinRBXRechitR45Cuts_;

  // "high level" cuts
  // used to determine where an RBX fails the high level noise cuts
  double hlMaxHPDEMF_;  // maximum HPD EMF
  double hlMaxRBXEMF_;  // maximum RBX EMF
};

class JoinCaloTowerRefVectorsWithoutDuplicates {
public:
  JoinCaloTowerRefVectorsWithoutDuplicates() {}
  ~JoinCaloTowerRefVectorsWithoutDuplicates() {}

  void operator()(edm::RefVector<CaloTowerCollection> &v1, const edm::RefVector<CaloTowerCollection> &v2) const;

private:
  // helper function to compare calotower references
  struct twrrefcomp {
    inline bool operator()(const edm::Ref<CaloTowerCollection> &t1, const edm::Ref<CaloTowerCollection> &t2) const {
      return t1->id() < t2->id();
    }
  };
  typedef std::set<edm::Ref<CaloTowerCollection>, twrrefcomp> twrrefset_t;
};

#endif
