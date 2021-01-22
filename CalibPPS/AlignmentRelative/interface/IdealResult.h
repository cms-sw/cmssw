/****************************************************************************
* Authors: 
*  Jan Kašpar (jan.kaspar@gmail.com) 
****************************************************************************/

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSGeometry.h"

#include "CondFormats/PPSObjects/interface/CTPPSRPAlignmentCorrectionData.h"
#include "CondFormats/PPSObjects/interface/CTPPSRPAlignmentCorrectionsData.h"

#include "CalibPPS/AlignmentRelative/interface/AlignmentAlgorithm.h"

#include <vector>

class AlignmentTask;

/**
 *\brief Calculates the ideal result of the StraightTrackAlignment.
 **/
class IdealResult : public AlignmentAlgorithm {
protected:
  edm::ESHandle<CTPPSGeometry> gReal, gMisaligned;

public:
  /// dummy constructor (not to be used)
  IdealResult() {}

  /// normal constructor
  IdealResult(const edm::ParameterSet &ps, AlignmentTask *_t);

  ~IdealResult() override {}

  std::string getName() override { return "Ideal"; }

  bool hasErrorEstimate() override { return false; }

  void begin(const CTPPSGeometry *geometryReal, const CTPPSGeometry *geometryMisaligned) override;

  void feed(const HitCollection &, const LocalTrackFit &) override {}

  void saveDiagnostics(TDirectory *) override {}

  void analyze() override {}

  unsigned int solve(const std::vector<AlignmentConstraint> &,
                     std::map<unsigned int, AlignmentResult> &result,
                     TDirectory *dir) override;

  void end() override {}
};
