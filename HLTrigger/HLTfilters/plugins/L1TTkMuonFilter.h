#ifndef L1TTkMuonFilter_h
#define L1TTkMuonFilter_h

/** \class L1TTkMuonFilter
 *
 *
 *  This class is an HLTFilter (-> EDFilter) implementing a very basic
 *  HLT trigger acting on TkMuon candidates
 *  2022-08-01: moving from TkMuon to TrackerMuon
 *
 *
 *  \author Simone Gennai
 *  \author Thiago Tomei
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "DataFormats/L1TMuonPhase2/interface/TrackerMuon.h"

//
// class declaration
//

class L1TTkMuonFilter : public HLTFilter {
public:
  explicit L1TTkMuonFilter(const edm::ParameterSet&);
  ~L1TTkMuonFilter() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  bool hltFilter(edm::Event&,
                 const edm::EventSetup&,
                 trigger::TriggerFilterObjectWithRefs& filterproduct) const override;

private:
  edm::InputTag l1TkMuonTag_;                                 //input tag for L1 TrackerMuon product
  edm::EDGetTokenT<l1t::TrackerMuonCollection> tkMuonToken_;  // token identifying product containing L1 TrackerMuons

  double min_Pt_;                        // min pt cut
  int min_N_;                            // min number of candidates above pT cut
  double min_Eta_;                       // min eta cut
  double max_Eta_;                       // max eta cut
  bool applyQuality_;                    // apply quality cuts
  bool applyDuplicateRemoval_;           // apply duplicate removal
  std::vector<int> qualities_;           // allowed qualities
  edm::ParameterSet scalings_;           // all scalings. An indirection level allows extra flexibility
  std::vector<double> barrelScalings_;   // barrel scalings
  std::vector<double> overlapScalings_;  // overlap scalings
  std::vector<double> endcapScalings_;   // endcap scalings

  double TkMuonOfflineEt(double Et, double Eta) const;
};

#endif  //L1TTkMuonFilter_h
