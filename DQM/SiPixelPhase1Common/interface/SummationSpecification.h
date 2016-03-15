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
// TODO: terminology is a bit messed up, and the code is not very clean, but
// there is really not much happening here except for input validation.
// 
// Original Author:  Marcel Schneider

#include <vector>
#include <set>
#include <string>
#include <cassert>

// there does not seem to be a standard string tokenizer...
#include <boost/algorithm/string.hpp>

#include "DQM/SiPixelPhase1Common/interface/GeometryInterface.h"

struct SummationStep {
  // For step1, all the necessary information should be in the type and columns
  // to allow fill() to exectute it very quickly.
  // For step2 stuff (after the first SAVE), we can also keep strings, since 
  // step2 will only be executed once by an executor.
  enum Type  {NO_TYPE, GROUPBY, EXTEND_X, EXTEND_Y, COUNT, SAVE};
  Type type = NO_TYPE;
  enum Stage {NO_STAGE, FIRST, STAGE1, STAGE2};
  Stage stage = NO_STAGE;

  std::vector<GeometryInterface::Column> columns;
  
  // more parameters. Burning them into the step-functor as a closure seems 
  // nice but we might want to have them exponsed, since this is not a program
  // that is just executed sequentially. It is a specification...
  std::string more_args;
};

struct SummationSpecification {
  std::vector<SummationStep> steps;
};

// The builder gets the empty spec passed in,then a chain of methods is called 
// to add instructions to the spec. It should always return itself and extend 
// the spec. If necessary, it parses the strings passed in.
// For step1, it might also convert the command to be processed easyly.
struct SummationSpecificationBuilder {
  SummationSpecification& spec;
  // small state machine to check validity of the program.
  SummationStep::Stage state = SummationStep::FIRST;
  std::set<GeometryInterface::Column> activeColums;

  SummationSpecificationBuilder(SummationSpecification& s) : spec(s) {};

  SummationSpecificationBuilder& groupBy(const char* cols, const char* mode = "SUM") {
    std::vector<std::string> cs;
    boost::split(cs, cols, boost::is_any_of("/"), boost::token_compress_on);
    auto step = SummationStep();
    step.stage = state;
    step.columns = cs;
    auto modename = std::string(mode);
    if (modename == "SUM") {
      step.type = SummationStep::GROUPBY;
      activeColums.clear();
      activeColums.insert(cs.begin(), cs.end());
    } else if (modename == "EXTEND_X" || modename == "EXTEND_Y") {
      assert(state != SummationStep::FIRST || !"First statement must have SUM summation.");
      // use set diff to find removed column and store only that.
      for (auto x : cs) activeColums.erase(x);
      if (activeColums.size() > 1 || activeColums.size() == 0) {
	//TODO: Better logging?
        assert(!"groupBy with extension has to drop exactly on column");
      }
      step.columns.push_back(*activeColums.begin()); // precisely one el.
      activeColums.clear();
      activeColums.insert(cs.begin(), cs.end());
      if (modename == "EXTEND_X") step.type = SummationStep::EXTEND_X;
      else                    step.type = SummationStep::EXTEND_Y;
    } else {
      assert(!"Unsupported summation mode.");
    }

    if (state == SummationStep::FIRST) state = SummationStep::STAGE1;
    spec.steps.push_back(step);
    return *this;
  }

  SummationSpecificationBuilder& save() {
    assert(state != SummationStep::FIRST || !"First statement must be groupBy.");
    auto step = SummationStep();
    step.stage = state;
    step.type = SummationStep::SAVE;
    state = SummationStep::STAGE2;
    spec.steps.push_back(step);
    return *this;
  }

  SummationSpecificationBuilder& count() {
    assert(state != SummationStep::FIRST || !"First statement must be groupBy.");
    auto step = SummationStep();
    step.stage = state;
    step.type = SummationStep::COUNT;
    spec.steps.push_back(step);
    return *this;
  }
  // TODO: step2 

  SummationSpecificationBuilder& reduce(std::string sort) {
    //TODO
    return *this;
  }

  SummationSpecificationBuilder& saveAll() {
    //TODO
    return save();
  }

};


#endif
