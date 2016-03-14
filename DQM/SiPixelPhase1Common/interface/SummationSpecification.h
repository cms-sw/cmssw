#ifndef SiPixel_SummationSpecification
#define SiPixel_SummationSpecification
// -*- C++ -*-
//
// Package:    SiPixelPhase1Common
// Class:      SummationSpecification
//
// This class represents a sequence of steps that produce histograms by summing
// up other histograms. This can be considered a domain-specific language for 
// DQM. This class has no intelligence, it just manages the "program". It is
// not encapsulated, the structure is exposed.
// 
// Original Author:  Marcel Schneider

#include <vector>
#include <set>
#include <string>

// there does not seem to be a standard string tokenizer...
#include <boost/algorithm/string.hpp>

#include "DQM/SiPixelPhase1Common/interface/GeometryInterface.h"

struct SummationStep {
  // TODO: this has to be worked out
  // std::function step;

  std::vector<GeometryInterface::Column> columns;
  
  // more parameters. Burning them into the step-functor as a closure seems 
  // nice but we might want to have them exponsed, since this is not a program
  // that is just executed sequentially. It is a specification...}
};

struct SummationSpecification {
  std::vector<SummationStep> steps;
};

// The builder gets the empty spec passed in,then a chain of methods is called 
// to add instructions to the spec. It should always return itself and extend 
// the spec. If necessary, it parses the strings passed in.
struct SummationSpecificationBuilder {
  SummationSpecification& spec;

  SummationSpecificationBuilder(SummationSpecification& s) : spec(s) {};

  SummationSpecificationBuilder groupBy(const char* cols) {
    std::vector<std::string> cs;
    boost::split(cs, cols, boost::is_any_of("/"));
    auto step = SummationStep();
    step.columns = cs;
    spec.steps.push_back(step);
    return *this;
  }
};


#endif
