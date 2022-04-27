#ifndef HLTGenericFilter_h
#define HLTGenericFilter_h

/** \class HLTGenericFilter
 *
 *  \author Roberto Covarelli (CERN)
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"

#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateIsolation.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

namespace edm {
  class ConfigurationDescriptions;
}

//
// class declaration
//

template <typename T1>
class HLTGenericFilter : public HLTFilter {
  typedef std::vector<T1> T1Collection;
  typedef edm::Ref<T1Collection> T1Ref;
  typedef edm::AssociationMap<edm::OneToValue<std::vector<T1>, float> > T1IsolationMap;

public:
  explicit HLTGenericFilter(const edm::ParameterSet&);
  ~HLTGenericFilter() override;
  bool hltFilter(edm::Event&,
                 const edm::EventSetup&,
                 trigger::TriggerFilterObjectWithRefs& filterproduct) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  float getEnergy(T1Ref) const;
  float getEt(T1Ref) const;

  edm::InputTag candTag_;  // input tag identifying product that contains filtered candidates
  edm::InputTag varTag_;   // input tag identifying product that contains variable map
  edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> candToken_;
  edm::EDGetTokenT<T1IsolationMap> varToken_;

  std::vector<double> energyLowEdges_;  // lower bin edges for energy-dependent cuts
  bool lessThan_;                       // the cut is "<" or ">" ?
  bool useEt_;                          // use E or Et in relative isolation cuts
  std::vector<double> thrRegularEB_;    // threshold for regular cut (x < thr) - ECAL barrel
  std::vector<double> thrRegularEE_;    // threshold for regular cut (x < thr) - ECAL endcap
  std::vector<double> thrOverEEB_;      // threshold for x/E < thr cut (isolations) - ECAL barrel
  std::vector<double> thrOverEEE_;      // threshold for x/E < thr cut (isolations) - ECAL endcap
  std::vector<double> thrOverE2EB_;     // threshold for x/E^2 < thr cut (isolations) - ECAL barrel
  std::vector<double> thrOverE2EE_;     // threshold for x/E^2 < thr cut (isolations) - ECAL endcap
  int ncandcut_;                        // number of candidates required

  edm::InputTag l1EGTag_;

  edm::InputTag rhoTag_;  // input tag identifying rho producer
  edm::EDGetTokenT<double> rhoToken_;
  bool doRhoCorrection_;
  double rhoMax_;
  double rhoScale_;
  std::vector<double> effectiveAreas_;
  std::vector<double> absEtaLowEdges_;
};

#endif  //HLTGenericFilter_h
