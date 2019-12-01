#ifndef __RecHitTopologicalCleanerBase_H__
#define __RecHitTopologicalCleanerBase_H__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"

#include <string>

class RecHitTopologicalCleanerBase {
public:
  RecHitTopologicalCleanerBase(const edm::ParameterSet& conf) {}
  RecHitTopologicalCleanerBase(const RecHitTopologicalCleanerBase&) = delete;
  virtual ~RecHitTopologicalCleanerBase() = default;
  RecHitTopologicalCleanerBase& operator=(const RecHitTopologicalCleanerBase&) = delete;

  virtual void clean(const edm::Handle<reco::PFRecHitCollection>&, std::vector<bool>&) = 0;
  virtual void update(const edm::EventSetup&) {}

  const std::string& name() const { return _algoName; }

private:
  const std::string _algoName;
};

#include "FWCore/PluginManager/interface/PluginFactory.h"
typedef edmplugin::PluginFactory<RecHitTopologicalCleanerBase*(const edm::ParameterSet&)>
    RecHitTopologicalCleanerFactory;

#endif
