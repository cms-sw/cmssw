#ifndef HLTRFilter_h
#define HLTRFilter_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METCollection.h"
#include "TVector3.h"
#include "TLorentzVector.h"

namespace edm {
  class ConfigurationDescriptions;
}

//
// class declaration
//

class HLTRFilter : public HLTFilter {
public:
  explicit HLTRFilter(const edm::ParameterSet&);
  ~HLTRFilter() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  bool hltFilter(edm::Event&,
                 const edm::EventSetup&,
                 trigger::TriggerFilterObjectWithRefs& filterproduct) const override;

  static double CalcMR(TLorentzVector ja, TLorentzVector jb);
  static double CalcR(double MR,
                      TLorentzVector ja,
                      TLorentzVector jb,
                      edm::Handle<edm::View<reco::MET>> met,
                      const std::vector<math::XYZTLorentzVector>& muons);
  //adds the values of MR and Rsq to the event as MET objects
  void addObjects(edm::Event&, trigger::TriggerFilterObjectWithRefs& filterproduct, double MR, double Rsq) const;

private:
  edm::EDGetTokenT<std::vector<math::XYZTLorentzVector>> m_theInputToken;
  edm::EDGetTokenT<edm::View<reco::MET>> m_theMETToken;
  edm::InputTag inputTag_;     // input tag identifying product
  edm::InputTag inputMetTag_;  // input tag identifying MET product
  bool doMuonCorrection_;      // do the muon corrections
  double min_R_;               // minimum R vaule
  double min_MR_;              // minimum MR vaule
  bool DoRPrime_;              // Do the R' instead of R
  bool accept_NJ_;             // accept or reject events with high NJ
  double R_offset_;            // R offset for parameterized cut
  double MR_offset_;           // MR offset for parameterized cut
  double R_MR_cut_;            // Cut value for parameterized cut
};

#endif  //HLTRFilter_h
