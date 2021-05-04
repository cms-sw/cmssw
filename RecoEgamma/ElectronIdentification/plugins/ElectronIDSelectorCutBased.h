#ifndef ElectronIDSelectorCutBased_h
#define ElectronIDSelectorCutBased_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

#include "ClassBasedElectronID.h"
#include "CutBasedElectronID.h"
#include "ElectronIDAlgo.h"
#include "PTDRElectronID.h"

class ElectronIDSelectorCutBased {
public:
  explicit ElectronIDSelectorCutBased(const edm::ParameterSet& conf, edm::ConsumesCollector&& iC)
      : ElectronIDSelectorCutBased(conf, iC) {}
  explicit ElectronIDSelectorCutBased(const edm::ParameterSet& conf, edm::ConsumesCollector& iC);
  virtual ~ElectronIDSelectorCutBased();

  void newEvent(const edm::Event&, const edm::EventSetup&);
  double operator()(const reco::GsfElectron&, const edm::Event&, const edm::EventSetup&);

private:
  ElectronIDAlgo* electronIDAlgo_;
  edm::ParameterSet conf_;
  std::string algorithm_;
};

#endif
