#ifndef HLTEgammaGenericQuadraticEtaFilter_h
#define HLTEgammaGenericQuadraticEtaFilter_h

/** \class HLTEgammaGenericQuadraticEtaFilter
 *
 *  \author Roberto Covarelli (CERN)
 *  modified by Chris Tully (Princeton)
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"

namespace edm {
  class ConfigurationDescriptions;
}

//
// class declaration
//

class HLTEgammaGenericQuadraticEtaFilter : public HLTFilter {
public:
  explicit HLTEgammaGenericQuadraticEtaFilter(const edm::ParameterSet&);
  ~HLTEgammaGenericQuadraticEtaFilter() override;
  bool hltFilter(edm::Event&,
                 const edm::EventSetup&,
                 trigger::TriggerFilterObjectWithRefs& filterproduct) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::InputTag candTag_;  // input tag identifying product that contains filtered candidates
  edm::InputTag varTag_;   // input tag identifying product that contains the variable map
  edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> candToken_;
  edm::EDGetTokenT<reco::RecoEcalCandidateIsolationMap> varToken_;
  bool lessThan_;  // the cut is "<" or ">" ?
  bool useEt_;     // use E or Et in relative isolation cuts
  /*  Barrel quadratic threshold function:
      vali (<= or >=) thrRegularEB_ + (E or Et)*thrOverEEB_ + (E or Et)*(E or Et)*thrOverE2EB_
    Endcap quadratic threshold function:
      vali (<= or >=) thrRegularEE_ + (E or Et)*thrOverEEE_ + (E or Et)*(E or Et)*thrOverE2EE_
      */

  std::vector<double> energyLowEdges_;  // lower bin edges for energy-dependent cuts
  double etaBoundaryEB12_;              //eta Boundary between Regions 1 and 2 - ECAL barrel
  double etaBoundaryEE12_;              //eta Boundary between Regions 1 and 2 - ECAL endcap
  std::vector<double> thrRegularEB1_;   // threshold value for zeroth order term - ECAL barrel region 1
  std::vector<double> thrRegularEE1_;   // threshold value for zeroth order term - ECAL endcap region 1
  std::vector<double> thrOverEEB1_;     // coefficient for first order term - ECAL barrel region 1
  std::vector<double> thrOverEEE1_;     // coefficient for first order term - ECAL endcap region 1
  std::vector<double> thrOverE2EB1_;    // coefficient for second order term - ECAL barrel region 1
  std::vector<double> thrOverE2EE1_;    // coefficient for second order term - ECAL endcap region 1
  std::vector<double> thrRegularEB2_;   // threshold value for zeroth order term - ECAL barrel region 2
  std::vector<double> thrRegularEE2_;   // threshold value for zeroth order term - ECAL endcap region 2
  std::vector<double> thrOverEEB2_;     // coefficient for first order term - ECAL barrel region 2
  std::vector<double> thrOverEEE2_;     // coefficient for first order term - ECAL endcap region 2
  std::vector<double> thrOverE2EB2_;    // coefficient for second order term - ECAL barrel region 2
  std::vector<double> thrOverE2EE2_;    // coefficient for second order term - ECAL endcap region 2
  int ncandcut_;                        // number of candidates required

  bool store_;
  edm::InputTag l1EGTag_;

  edm::InputTag rhoTag_;  // input tag identifying rho producer
  edm::EDGetTokenT<double> rhoToken_;
  bool doRhoCorrection_;
  double rhoMax_;
  double rhoScale_;
  std::vector<double> effectiveAreas_;
  std::vector<double> absEtaLowEdges_;
};

#endif  //HLTEgammaGenericQuadraticEtaFilter_h
