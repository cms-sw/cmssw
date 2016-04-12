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
#include <string>

#include "DQM/SiPixelPhase1Common/interface/GeometryInterface.h"

struct SummationStep {
  // For step1, all the necessary information should be in the type and columns
  // to allow fill() to exectute it very quickly.
  // For step2 stuff (after the first SAVE), we can also keep strings, since 
  // step2 will only be executed once by an executor.
  enum Type  {NO_TYPE, GROUPBY, EXTEND_X, EXTEND_Y, COUNT, REDUCE, SAVE, CUSTOM};
  Type type = NO_TYPE;
  enum Stage {NO_STAGE, FIRST, STAGE1, STAGE2};
  Stage stage = NO_STAGE;

  std::vector<GeometryInterface::Column> columns;
  
  // more parameters. Not very elegant but good enough for step2.
  std::string arg;
};

struct SummationSpecification {
  std::vector<SummationStep> steps;
  SummationSpecification() {};
  SummationSpecification(edm::ParameterSet const&, GeometryInterface&);

  template<class stream, class GI>
  void dump(stream& out, GI& gi) {
    for (auto& s : steps) {
      out << "Step: type " << s.type << " stage " << s.stage << " col ";
      for (auto c : s.columns) out << gi.pretty(c) << " ";
      out << " arg " << s.arg << "\n";
    }
  }
  private:
  GeometryInterface::Column parse_columns(std::string name, GeometryInterface&);
};

#endif
