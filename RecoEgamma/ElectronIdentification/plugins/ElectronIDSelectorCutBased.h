#ifndef ElectronIDSelectorCutBased_h
#define ElectronIDSelectorCutBased_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

#include "ElectronIDAlgo.h"

#include <memory>

class ElectronIDSelectorCutBased {
public:
  explicit ElectronIDSelectorCutBased(const edm::ParameterSet& conf, edm::ConsumesCollector&& iC)
      : ElectronIDSelectorCutBased(conf, iC) {}
  explicit ElectronIDSelectorCutBased(const edm::ParameterSet& conf, edm::ConsumesCollector& iC);
  virtual ~ElectronIDSelectorCutBased() = default;

  double operator()(const reco::GsfElectron&, const edm::Event&, const edm::EventSetup&) const;

private:
  std::unique_ptr<ElectronIDAlgo const> electronIDAlgo_;
};

#endif
