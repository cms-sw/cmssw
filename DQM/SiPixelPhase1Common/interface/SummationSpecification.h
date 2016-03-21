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
  enum Type  {NO_TYPE, GROUPBY, EXTEND_X, EXTEND_Y, COUNT, REDUCE, SAVE};
  Type type = NO_TYPE;
  enum Stage {NO_STAGE, FIRST, STAGE1, STAGE2};
  Stage stage = NO_STAGE;

  std::vector<GeometryInterface::Column> columns;
  
  // more parameters. Not very elegant but good enough for step2.
  std::string arg;
};

struct SummationSpecification {
  std::vector<SummationStep> steps;

  template<class stream>
  void dump(stream& out) {
    for (auto& s : steps) {
      out << "Step: type " << s.type << " stage " << s.stage << " col ";
      for (auto c : s.columns) out << c << " ";
      out << " arg " << s.arg << "\n";
    }
  }
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

  // General grouping, pass in the columns that should remain and the mode of 
  // histogram summation.
  SummationSpecificationBuilder& groupBy(const char* cols, const char* mode = "SUM");
  // Save the current state of the table as MonitorElements. Marks transition to step2.
  SummationSpecificationBuilder& save();
  // Reduce a higher-dimensional hisotgram to a lower (typ. single number) one.
  SummationSpecificationBuilder& reduce(std::string sort);
  SummationSpecificationBuilder& count(); // special case of reduce
  // Save all parents, summed up like in the last grouping, in the hierarchy 
  // as specified.
  SummationSpecificationBuilder& saveAll();
};


#endif
