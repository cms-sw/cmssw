#ifndef L1TPFTauFilter_h
#define L1TPFTauFilter_h

/** \class L1TPFTauFilter
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

class L1TPFTauFilter : public HLTFilter {
public:
  explicit L1TPFTauFilter(const edm::ParameterSet&);
  ~L1TPFTauFilter() override;
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

#endif  //L1TPFTauFilter_h
