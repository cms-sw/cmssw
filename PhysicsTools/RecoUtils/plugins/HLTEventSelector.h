#ifndef Workspace_HLTEventSelector_h_
#define Workspace_HLTEventSelector_h_

/** Trivial example for a HLT selector.
 *  To be modified for analysis!
 */
// Original author: W. Adam, 10/4/08

// system include files
#include <memory>

// user include files
#include "CommonTools/UtilAlgos/interface/EventSelector.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/TriggerResults.h"

#include <vector>
#include <string>

class HLTEventSelector : public EventSelector {
public:
  HLTEventSelector(const edm::ParameterSet& pset, edm::ConsumesCollector&& iC) : HLTEventSelector(pset, iC) {}
  HLTEventSelector(const edm::ParameterSet& pset, edm::ConsumesCollector& iC);
  bool select(const edm::Event&) const override;
  ~HLTEventSelector() override {}

private:
  edm::InputTag triggerResults_;  ///< tag for input collection
  edm::EDGetTokenT<edm::TriggerResults> triggerResultsToken_;
  std::vector<std::string> pathNames_;  ///< trigger path names (ORed)
};
#endif
