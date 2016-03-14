// -*- C++ -*-
//
// Package:     SiPixelPhase1Common
// Class  :     HistogramManager
//
#include "DQM/SiPixelPhase1Common/interface/HistogramManager.h"

HistogramManager::HistogramManager(const edm::ParameterSet& iconfig) :
  iConfig(iconfig)
{
}

void HistogramManager::addSpec(SummationSpecification& spec) {
  specs.push_back(spec);
}

SummationSpecificationBuilder HistogramManager::addSpec() {
  specs.push_back(SummationSpecification());
  return SummationSpecificationBuilder(specs[specs.size()-1]);
}

// note that this will be pretty hot. Ideally it should be malloc-free.
void HistogramManager::fill(double value, DetId sourceModule, edm::Event *sourceEvent, int col, int row) {
  if(!columsFinal) {
    for (SummationSpecification const& s : specs) {
      for (auto c : s.steps[0].columns) {
        significantColumns.insert(c);
      }
    }
    columsFinal = true;
  }

  auto significantvalues = geometryInterface.extractColumns(significantColumns, 
                             GeometryInterface::InterestingQuantities{
			       sourceModule, sourceEvent, col, row
			     });
  table[significantvalues].fill(value);
}
  
void HistogramManager::book(DQMStore::IBooker& iBooker, edm::EventSetup const& iSetup) {
  if (!geometryInterface.loaded()) {
    geometryInterface.load(iSetup, iConfig);
  }
  // TODO: more stuff here. 
}

