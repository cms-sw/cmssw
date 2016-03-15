// -*- C++ -*-
//
// Package:     SiPixelPhase1Common
// Class  :     HistogramManager
//
#include "DQM/SiPixelPhase1Common/interface/HistogramManager.h"

HistogramManager::HistogramManager(const edm::ParameterSet& iconfig) :
  iConfig(iconfig),
  topFolderName(iconfig.getParameter<std::string>("TopFolderName"))
{
}

void HistogramManager::addSpec(SummationSpecification spec) {
  specs.push_back(spec);
  tables.push_back(Table());
}

SummationSpecificationBuilder HistogramManager::addSpec() {
  addSpec(SummationSpecification());
  return SummationSpecificationBuilder(specs[specs.size()-1]);
}

// note that this will be pretty hot. Ideally it should be malloc-free.
void HistogramManager::fill(double value, DetId sourceModule, const edm::Event *sourceEvent, int col, int row) {
  auto iq = GeometryInterface::InterestingQuantities{
              sourceModule, sourceEvent, col, row
	    };							    
  for (unsigned int i = 0; i < specs.size(); i++) {
    auto& s = specs[i];
    auto& t = tables[i];
    auto significantvalues = geometryInterface.extractColumns(s.steps[0].columns, iq);
    // TODO: more step1 steps must be excuted here. Step1 steps can be applied per-sample.
    // Step1 steps are those that have s.stage = SummationStep::STAGE1, and step[0].
    t[significantvalues].fill(value);
  }

  
}
  
void HistogramManager::book(DQMStore::IBooker& iBooker, edm::EventSetup const& iSetup) {
  if (!geometryInterface.loaded()) {
    geometryInterface.load(iSetup, iConfig);
  }
  iBooker.setCurrentFolder(topFolderName);
  // TODO: more stuff here. 
}

