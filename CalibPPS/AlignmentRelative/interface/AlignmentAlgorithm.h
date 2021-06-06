/****************************************************************************
* Authors: 
*  Jan Kašpar (jan.kaspar@gmail.com) 
****************************************************************************/

#ifndef CalibPPS_AlignmentRelative_AlignmentAlgorithm_h
#define CalibPPS_AlignmentRelative_AlignmentAlgorithm_h

#include "FWCore/Framework/interface/EventSetup.h"

#include "CalibPPS/AlignmentRelative/interface/LocalTrackFit.h"
#include "CalibPPS/AlignmentRelative/interface/AlignmentGeometry.h"
#include "CalibPPS/AlignmentRelative/interface/HitCollection.h"
#include "CalibPPS/AlignmentRelative/interface/AlignmentConstraint.h"
#include "CalibPPS/AlignmentRelative/interface/AlignmentResult.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSGeometry.h"

#include <string>
#include <map>

class AlignmentTask;
class TDirectory;

namespace edm {
  class ParameterSet;
}

/**
 *\brief Abstract parent for all (track-based) alignment algorithms
 **/
class AlignmentAlgorithm {
protected:
  unsigned int verbosity;

  /// the tasked to be completed
  AlignmentTask *task;

  /// eigenvalues in (-singularLimit, singularLimit) are treated as singular
  double singularLimit;

public:
  /// dummy constructor (not to be used)
  AlignmentAlgorithm() {}

  /// normal constructor
  AlignmentAlgorithm(const edm::ParameterSet &ps, AlignmentTask *_t);

  virtual ~AlignmentAlgorithm() {}

  virtual std::string getName() { return "Base"; }

  /// returns whether this algorithm is capable of estimating result uncertainties
  virtual bool hasErrorEstimate() = 0;

  /// prepare for processing
  virtual void begin(const CTPPSGeometry *geometryReal, const CTPPSGeometry *geometryMisaligned) = 0;

  /// process one track
  virtual void feed(const HitCollection &, const LocalTrackFit &) = 0;

  /// saves diagnostic histograms/plots
  virtual void saveDiagnostics(TDirectory *) = 0;

  /// analyzes the data collected
  virtual void analyze() = 0;

  /// solves the alignment problem with the given constraints
  /// \param dir a directory (in StraightTrackAlignment::taskDataFileName) where
  /// intermediate results can be stored
  virtual unsigned int solve(const std::vector<AlignmentConstraint> &,
                             std::map<unsigned int, AlignmentResult> &results,
                             TDirectory *dir = nullptr) = 0;

  /// cleans up after processing
  virtual void end() = 0;
};

#endif
