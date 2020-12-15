#ifndef L1TTkEmFilter_h
#define L1TTkEmFilter_h

/** \class L1TTkEmFilter
 *
 *
 *  This class is an HLTFilter (-> EDFilter) implementing a very basic
 *  HLT trigger acting on TkEm candidates
 *  This has support for *two* collections, since photons can come
 *  either from crystal calo or HGCAL
 *
 *  \author Simone Gennai
 *  \author Thiago Tomei
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "DataFormats/L1TCorrelator/interface/TkEm.h"
#include "DataFormats/L1TCorrelator/interface/TkEmFwd.h"

//
// class declaration
//

class L1TTkEmFilter : public HLTFilter {
public:
  explicit L1TTkEmFilter(const edm::ParameterSet&);
  ~L1TTkEmFilter() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  bool hltFilter(edm::Event&,
                 const edm::EventSetup&,
                 trigger::TriggerFilterObjectWithRefs& filterproduct) const override;

private:
  edm::InputTag l1TkEmTag1_;  //input tag for L1 Tk Em product
  edm::InputTag l1TkEmTag2_;  //input tag for L1 Tk Em product

  typedef std::vector<l1t::TkEm> TkEmCollection;
  edm::EDGetTokenT<TkEmCollection> tkEmToken1_;  // token identifying product containing L1 TkEms
  edm::EDGetTokenT<TkEmCollection> tkEmToken2_;  // token identifying product containing L1 TkEms

  double min_Pt_;                            // min pt cut
  int min_N_;                                // min number of candidates above pT cut
  double min_Eta_;                           //min eta cut
  double max_Eta_;                           //max eta cut
  edm::ParameterSet scalings_;               // all scalings. An indirection level allows extra flexibility
  std::vector<double> barrelScalings_;       // barrel scalings
  std::vector<double> endcapScalings_;       // endcap scalings
  std::vector<double> etaBinsForIsolation_;  // abs. eta bin edges for isolation. First edge at 0.0 must be explicit!
  std::vector<double> trkIsolation_;         // values for track isolation in the bins defined above
  int quality1_;                             // quality for photons of 1st collection
  int quality2_;                             // quality for photons of 2nd collection
  int qual1IsMask_;                          // is qual for photons of 1st collection a mask?
  int qual2IsMask_;                          // is qual for photons of 2nd collection a mask?
  bool applyQual1_;                          // should we apply quality 1?
  bool applyQual2_;                          // should we apply quality 2?

  double TkEmOfflineEt(double Et, double Eta) const;
};

#endif  //L1TTkEmFilter_h
