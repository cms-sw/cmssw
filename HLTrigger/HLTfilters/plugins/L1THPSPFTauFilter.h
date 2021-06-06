#ifndef L1THPSPFTauFilter_h
#define L1THPSPFTauFilter_h

/** \class L1THPSPFTauFilter
 *
 *
 *  This class is an HLTFilter (-> EDFilter) implementing a very basic
 *  HLT trigger acting on HPSPFTau candidates
 *
 *
 *
 *  \author Sandeep Bhowmik
 *  \author Thiago Tomei
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "DataFormats/L1TParticleFlow/interface/HPSPFTau.h"
#include "DataFormats/L1TParticleFlow/interface/HPSPFTauFwd.h"

//
// class declaration
//

class L1THPSPFTauFilter : public HLTFilter {
public:
  explicit L1THPSPFTauFilter(const edm::ParameterSet&);
  ~L1THPSPFTauFilter() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  bool hltFilter(edm::Event&,
                 const edm::EventSetup&,
                 trigger::TriggerFilterObjectWithRefs& filterproduct) const override;

private:
  edm::InputTag l1HPSPFTauTag_;                              // input tag for L1 HPSPFTau product
  edm::EDGetTokenT<l1t::HPSPFTauCollection> hpspfTauToken_;  // token identifying product containing L1 HPSPFTaus
  double min_Pt_;                                            // min pt cut
  int min_N_;                                                // min number of candidates above pT cut
  double min_Eta_;                                           // min eta cut
  double max_Eta_;                                           // max eta cut
  double max_RelChargedIso_;                                 // max relative charged isolation
  double min_LeadTrackPt_;                                   // min leading track pT
  edm::ParameterSet scalings_;           // all scalings. An indirection level allows extra flexibility
  std::vector<double> barrelScalings_;   // barrel scalings
  std::vector<double> overlapScalings_;  // overlap scalings
  std::vector<double> endcapScalings_;   // endcap scalings

  double HPSPFTauOfflineEt(double Et, double Eta) const;
};

#endif  //L1THPSPFTauFilter_h
