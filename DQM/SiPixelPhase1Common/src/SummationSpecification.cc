// -*- C++ -*-
//
// Package:    SiPixelPhase1Common
// Class:      SummationSpecification
//
// SummationSpecification  does not need any impl, so this is mostly about the Builder.
//
// Original Author:  Marcel Schneider

#include "DQM/SiPixelPhase1Common/interface/SummationSpecification.h"

#include <set>
#include <cassert>

// there does not seem to be a standard string tokenizer...
#include <boost/algorithm/string.hpp>

SummationSpecificationBuilder&  
SummationSpecificationBuilder::groupBy(const char* cols, const char* mode) {
  std::vector<std::string> cs;
  boost::split(cs, cols, boost::is_any_of("/"), boost::token_compress_on);
  auto step = SummationStep();
  step.stage = state;
  auto modename = std::string(mode);
  if (modename == "SUM") {
    step.type = SummationStep::GROUPBY;
    step.columns = cs;
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

SummationSpecificationBuilder& 
SummationSpecificationBuilder::save() {
  assert(state != SummationStep::FIRST || !"First statement must be groupBy.");
  auto step = SummationStep();
  step.stage = state;
  step.type = SummationStep::SAVE;
  state = SummationStep::STAGE2;
  spec.steps.push_back(step);
  return *this;
}

SummationSpecificationBuilder& 
SummationSpecificationBuilder::reduce(std::string sort) {
  assert(state != SummationStep::FIRST || !"First statement must be groupBy.");
  auto step = SummationStep();
  step.stage = state;
  step.type = SummationStep::REDUCE;
  assert(sort == "MEAN" || sort == "COUNT"); 
  step.arg = sort;
  if (state == SummationStep::STAGE1) {
    // for step1, we don't want to look at the string arg.
    assert(sort == "COUNT");
    step.type = SummationStep::COUNT;
  }
  spec.steps.push_back(step);
  return *this;
}

// This is surprisingly complicated, since we don't have the user input at hand,
// and the iternal repr is a bit more complicated.
SummationSpecificationBuilder& 
SummationSpecificationBuilder::saveAll() {
  // first we need the last grouping, to know the summation mode
  int i;
  for (i = spec.steps.size()-1; i >= 0; i--) {
    if (spec.steps[i].type == SummationStep::GROUPBY
     || spec.steps[i].type == SummationStep::EXTEND_X 
     || spec.steps[i].type == SummationStep::EXTEND_Y) break;
  }
  assert (i >= 0 || !"No matching groupBy found.");
  auto& s = spec.steps[i];

  // now we go through the columns in reverse order and add a groupby for the
  // cols up to i. However, some cols might already be reduced (we cant use 
  // activeColums directly since we need the original order) and depending on
  // the summation type we need different formats for the new step.
  auto allcols = spec.steps[0].columns;
  for (int i = allcols.size()-1; i >= 0; i--) {
    // dont' forget to actually save
    save();
    auto c = activeColums.find(allcols.at(i));
    if (c != activeColums.end()) {
      if (s.type == SummationStep::GROUPBY) {
	// GROUPBY (all columns in arg)
	auto step = SummationStep();
	step.stage = state;
	step.columns = std::vector<GeometryInterface::Column>(allcols);
	step.columns.erase(step.columns.begin() + i, step.columns.end());
	step.type = s.type;
	spec.steps.push_back(step);
      } else {
	// EXTEND step (dropped col in arg)
	auto step = SummationStep();
	step.stage = state;
	step.columns.push_back(*c);
	step.type = s.type;
	spec.steps.push_back(step);
      }
      activeColums.erase(c);
    }
  }
  save();
  return *this;
}
