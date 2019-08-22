#ifndef HLTEgammaDoubleEtPhiFilter_h
#define HLTEgammaDoubleEtPhiFilter_h

/** \class HLTEgammaDoubleEtPhiFilter
 *
 *  \author Jonathan Hollar (LLNL)
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefToBase.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

namespace edm {
  class ConfigurationDescriptions;
}

//
// class decleration
//

class HLTEgammaDoubleEtPhiFilter : public HLTFilter {
public:
  explicit HLTEgammaDoubleEtPhiFilter(const edm::ParameterSet&);
  ~HLTEgammaDoubleEtPhiFilter() override;
  bool hltFilter(edm::Event&,
                 const edm::EventSetup&,
                 trigger::TriggerFilterObjectWithRefs& filterproduct) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::InputTag candTag_;  // input tag identifying product contains filtered candidates
  edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> candToken_;
  double etcut1_;         // Et threshold in GeV
  double etcut2_;         // Et threshold in GeV
  double min_Acop_;       // minimum acoplanarity
  double max_Acop_;       // maximum acoplanarity
  double min_EtBalance_;  // minimum Et difference
  double max_EtBalance_;  // maximum Et difference
  int npaircut_;          // number of egammas required
};

#endif  //HLTEgammaDoubleEtPhiFilter_h
