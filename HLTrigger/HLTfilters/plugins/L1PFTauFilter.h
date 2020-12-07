#ifndef L1PFTauFilter_h
#define L1PFTauFilter_h

/** \class L1PFTauFilter
 *
 *
 *  This class is an HLTFilter (-> EDFilter) implementing a very basic
 *  HLT trigger acting on PFTau (NN) candidates
 *
 *
 *
 *  \author Thiago Tomei
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "DataFormats/L1TParticleFlow/interface/PFTau.h"

//
// class declaration
//

class L1PFTauFilter : public HLTFilter {
public:
  explicit L1PFTauFilter(const edm::ParameterSet&);
  ~L1PFTauFilter() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  bool hltFilter(edm::Event&,
                 const edm::EventSetup&,
                 trigger::TriggerFilterObjectWithRefs& filterproduct) const override;

private:
  edm::InputTag l1PFTauTag_;                           //input tag for L1 PFTau product
  edm::EDGetTokenT<l1t::PFTauCollection> pfTauToken_;  // token identifying product containing L1 PFTaus
  double min_Pt_;                                      // min pt cut
  int min_N_;                                          // min number of candidates above pT cut
  double min_Eta_;                                     //min eta cut
  double max_Eta_;                                     //max eta cut
  double maxChargedIso_;                               // Cut on charged isolation
  double maxFullIso_;                                  // Cut on full isolation
  int passLooseNN_;                                    // Pass loose NN cut... for some implementation of the NN
  int passTightNN_;                                    // Pass tight NN cut... for some implementation of the NN
  edm::ParameterSet scalings_;                         // all scalings. An indirection level allows extra flexibility
  std::vector<double> barrelScalings_;                 // barrel scalings
  std::vector<double> endcapScalings_;                 // endcap scalings

  double PFTauOfflineEt(double Et, double Eta) const;
};

#endif  //L1PFTauFilter_h
