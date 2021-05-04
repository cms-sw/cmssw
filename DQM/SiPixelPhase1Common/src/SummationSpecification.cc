// -*- C++ -*-
//
// Package:    SiPixelPhase1Common
// Class:      SummationSpecification
//
// SummationSpecification  does not need much impl, mostly the constructor.
//
// Original Author:  Marcel Schneider

#include "DQM/SiPixelPhase1Common/interface/SummationSpecification.h"

#include <set>
#include <cassert>

GeometryInterface::Column SummationSpecification::parse_columns(std::string name,
                                                                GeometryInterface& geometryInterface) {
  return geometryInterface.intern(name);
}

SummationSpecification::SummationSpecification(const edm::ParameterSet& config, GeometryInterface& geometryInterface) {
  auto spec = config.getParameter<edm::VParameterSet>("spec");

  for (const auto& step : spec) {
    auto s = SummationStep();
    s.type = SummationStep::Type(step.getParameter<int>("type"));
    s.stage = SummationStep::Stage(step.getParameter<int>("stage"));

    s.nbins = int(step.getParameter<int>("nbins"));
    s.xmin = int(step.getParameter<int>("xmin"));
    s.xmax = int(step.getParameter<int>("xmax"));

    for (const auto& c : step.getParameter<std::vector<std::string>>("columns")) {
      s.columns.push_back(parse_columns(c, geometryInterface));
    }
    s.arg = step.getParameter<std::string>("arg");
    steps.push_back(s);
  }
}
