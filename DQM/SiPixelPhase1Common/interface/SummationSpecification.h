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
#include <string>

#include "DQM/SiPixelPhase1Common/interface/GeometryInterface.h"

struct SummationStep {
  // For step1, all the necessary information should be in the type and columns
  // to allow fill() to exectute it very quickly.
  // For step2 stuff (after the first SAVE), we can also keep strings, since
  // step2 will only be executed once by an executor.
  enum Type {
    NO_TYPE = 0,
    GROUPBY = 1,
    EXTEND_X = 2,
    EXTEND_Y = 3,
    COUNT = 4,
    REDUCE = 5,
    SAVE = 6,
    USE_X = 8,
    USE_Y = 9,
    USE_Z = 10,
    PROFILE = 11
  };
  Type type = NO_TYPE;
  // STAGE1 is DQM step1, STAGE2 step2. STAGE1_2 is somewhere in between, it runs
  // in the analyze()-method (step1) but does a sort of harvesting (per-event).
  // STAGE1_2 is for ndigis-like counters.
  // FIRST is the first group-by, which is special.
  enum Stage { NO_STAGE, FIRST, STAGE1, STAGE2 };
  Stage stage = NO_STAGE;

  int nbins{-1};
  int xmin{0};
  int xmax{0};

  std::vector<GeometryInterface::Column> columns;

  // more parameters. Not very elegant but good enough for step2.
  std::string arg;
};

struct SummationSpecification {
  std::vector<SummationStep> steps;
  SummationSpecification(){};
  SummationSpecification(edm::ParameterSet const&, GeometryInterface&);

  template <class stream, class GI>
  void dump(stream& out, GI& gi) {
    for (auto& s : steps) {
      out << "Step: type " << s.type << " stage " << s.stage << " col ";
      for (auto c : s.columns)
        out << gi.pretty(c) << " ";
      out << " arg " << s.arg << "\n";
    }
  }

private:
  GeometryInterface::Column parse_columns(std::string name, GeometryInterface&);
};

#endif
