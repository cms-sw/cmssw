#ifndef HLTPFEnergyFractionsFilter_h
#define HLTPFEnergyFractionsFilter_h

/** \class HLTPFEnergyFractionsFilter
 *
 *  \author Srimanobhas Phat
 *
 *  This filter is used to filter the PFJet collection using JetID.
 *  If you want to work with general PFJet collection, please use PFJetIDProducer instead.
 *
 *  This filter will reject event in 2 cases, 
 *   (1) No. of jets < NJet_ threshold. The default is 1 (You really need to be careful this setting).
 *   (2) One or more first NJet_ jets has PFEFs out of thresholds.
 *
 *  Just to warn you that, analyzers need to understand well their signals and percentage of loss if you apply this cut.
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"

namespace edm {
  class ConfigurationDescriptions;
}

//
// class declaration
//

class HLTPFEnergyFractionsFilter : public HLTFilter {
public:
  explicit HLTPFEnergyFractionsFilter(const edm::ParameterSet&);
  ~HLTPFEnergyFractionsFilter() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  bool hltFilter(edm::Event&,
                 const edm::EventSetup&,
                 trigger::TriggerFilterObjectWithRefs& filterproduct) const override;

private:
  edm::EDGetTokenT<reco::PFJetCollection> m_thePFJetToken;
  edm::InputTag inputPFJetTag_;  // input tag identifying pfjets
  unsigned int nJet_;            // No. of jet to check with this filter
  double min_CEEF_;
  double max_CEEF_;
  double min_NEEF_;
  double max_NEEF_;
  double min_CHEF_;
  double max_CHEF_;
  double min_NHEF_;
  double max_NHEF_;
  int triggerType_;
};

#endif  //HLTPFEnergyFractionsFilter_h
