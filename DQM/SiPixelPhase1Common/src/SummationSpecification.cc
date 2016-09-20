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

// there does not seem to be a standard string tokenizer...
#include <boost/algorithm/string.hpp>

GeometryInterface::Column 
SummationSpecification::parse_columns(std::string name, GeometryInterface& geometryInterface) {
  std::vector<std::string> parts;
  boost::split(parts, name, boost::is_any_of("|"), boost::token_compress_on);
  GeometryInterface::Column out = {{0,0}};
  unsigned int i = 0;
  for (auto str : parts) {
    assert(i < out.size() || !"maximum number of alternative columns exceeded");
    out[i++] = geometryInterface.intern(str); 
  }
  return out;
}


SummationSpecification::SummationSpecification(const edm::ParameterSet& config, GeometryInterface& geometryInterface) {
  auto spec = config.getParameter<edm::VParameterSet>("spec");
  for (auto step : spec) {
    auto s = SummationStep();
    s.type = SummationStep::Type(step.getParameter<int>("type"));
    s.stage = SummationStep::Stage(step.getParameter<int>("stage"));
    for (auto c : step.getParameter<std::vector<std::string>>("columns")) {
      s.columns.push_back(parse_columns(c, geometryInterface));
    }
    s.arg = step.getParameter<std::string>("arg");
    steps.push_back(s);
  }
}


