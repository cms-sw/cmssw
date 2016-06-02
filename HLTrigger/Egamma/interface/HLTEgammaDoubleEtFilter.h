#ifndef HLTEgammaDoubleEtFilter_h
#define HLTEgammaDoubleEtFilter_h

/** \class HLTEgammaDoubleEtFilter
 *
 *  \author Monica Vazquez Acosta (CERN)
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

class HLTEgammaDoubleEtFilter : public HLTFilter {

 public:
  explicit HLTEgammaDoubleEtFilter(const edm::ParameterSet&);
  ~HLTEgammaDoubleEtFilter();
  virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

 private:
  edm::InputTag candTag_; // input tag identifying product contains filtered candidates
  edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs>  candToken_;
  double etcut1_;           // Et threshold in GeV
  double etcut2_;           // Et threshold in GeV
  int    npaircut_;        // number of egammas required

  edm::InputTag l1EGTag_;
};

#endif //HLTEgammaDoubleEtFilter_h
