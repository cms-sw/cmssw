#ifndef L1TTkEleFilter_h
#define L1TTkEleFilter_h

/** \class L1TTkEleFilter
 *
 *
 *  This class is an HLTFilter (-> EDFilter) implementing a very basic
 *  HLT trigger acting on TkEle candidates
 *  This has support for *two* collections, since electrons can come
 *  either from crystal calo or HGCAL
 *
 *  \author Simone Gennai
 *  \author Thiago Tomei
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "DataFormats/L1TCorrelator/interface/TkElectron.h"
#include "DataFormats/L1TCorrelator/interface/TkElectronFwd.h"

//
// class declaration
//

class L1TTkEleFilter : public HLTFilter {
public:
  explicit L1TTkEleFilter(const edm::ParameterSet&);
  ~L1TTkEleFilter() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  bool hltFilter(edm::Event&,
                 const edm::EventSetup&,
                 trigger::TriggerFilterObjectWithRefs& filterproduct) const override;

private:
  edm::InputTag l1TkEleTag1_;  //input tag for L1 Tk Ele product
  edm::InputTag l1TkEleTag2_;  //input tag for L1 Tk Ele product

  typedef std::vector<l1t::TkElectron> TkEleCollection;
  edm::EDGetTokenT<TkEleCollection> tkEleToken1_;  // token identifying product containing L1 TkEles
  edm::EDGetTokenT<TkEleCollection> tkEleToken2_;  // token identifying product containing L1 TkEles

  double min_Pt_;                            // min pt cut
  int min_N_;                                // min number of candidates above pT cut
  double min_Eta_;                           // min eta cut
  double max_Eta_;                           // max eta cut
  edm::ParameterSet scalings_;               // all scalings. An indirection level allows extra flexibility
  std::vector<double> barrelScalings_;       // barrel scalings
  std::vector<double> endcapScalings_;       // endcap scalings
  std::vector<double> etaBinsForIsolation_;  // abs. eta bin edges for isolation. First edge at 0.0 must be explicit!
  std::vector<double> trkIsolation_;         // values for track isolation in the bins defined above
  int quality1_;                             // quality for electrons of 1st collection
  int quality2_;                             // quality for electrons of 2nd collection
  int qual1IsMask_;                          // is qual for electrons of 1st collection a mask?
  int qual2IsMask_;                          // is qual for electrons of 2nd collection a mask?
  bool applyQual1_;                          // should we apply quality 1?
  bool applyQual2_;                          // should we apply quality 2?

  double TkEleOfflineEt(double Et, double Eta) const;
};

#endif  //L1TTkEleFilter_h
