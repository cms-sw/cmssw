#ifndef L1TkMuonFilter_h
#define L1TkMuonFilter_h

/** \class L1TkMuonFilter
 *
 *
 *  This class is an HLTFilter (-> EDFilter) implementing a very basic
 *  HLT trigger acting on TkMuon candidates
 *
 *
 *
 *  \author Simone Gennai
 *  \author Thiago Tomei
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "DataFormats/L1TCorrelator/interface/TkMuon.h"
#include "DataFormats/L1TCorrelator/interface/TkMuonFwd.h"

//
// class declaration
//

class L1TkMuonFilter : public HLTFilter {
public:
  explicit L1TkMuonFilter(const edm::ParameterSet&);
  ~L1TkMuonFilter() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  bool hltFilter(edm::Event&,
                 const edm::EventSetup&,
                 trigger::TriggerFilterObjectWithRefs& filterproduct) const override;

private:
  edm::InputTag l1TkMuonTag_;  //input tag for L1 Tk Muon product
  typedef std::vector<l1t::TkMuon> TkMuonCollection;
  edm::EDGetTokenT<TkMuonCollection> tkMuonToken_;  // token identifying product containing L1 TkMuons

  double min_Pt_;                        // min pt cut
  int min_N_;                            // min number of candidates above pT cut
  double min_Eta_;                       // min eta cut
  double max_Eta_;                       // max eta cut
  edm::ParameterSet scalings_;           // all scalings. An indirection level allows extra flexibility
  std::vector<double> barrelScalings_;   // barrel scalings
  std::vector<double> overlapScalings_;  // overlap scalings
  std::vector<double> endcapScalings_;   // endcap scalings

  double TkMuonOfflineEt(double Et, double Eta) const;
};

#endif  //L1TkMuonFilter_h
