#ifndef NTupler_H
#define NTupler_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ProducesCollector.h"

#include "FWCore/Framework/interface/Event.h"
#include "TTree.h"

/*
 * Description:
 * placeholder for common ntuplizer tools
 *
 */

//base generic class

class NTupler {
public:
  NTupler() : useTFileService_(false) {}
  virtual ~NTupler() {}

  virtual unsigned int registerleaves(edm::ProducesCollector) = 0;
  virtual void fill(edm::Event& iEvent) = 0;

protected:
  bool useTFileService_;
  TTree* tree_;
};

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

typedef edmplugin::PluginFactory<NTupler*(const edm::ParameterSet&)> NTuplerFactory;

#endif
