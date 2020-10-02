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

  virtual ~IdealResult() {}

  virtual std::string getName() override { return "Ideal"; }

  virtual bool hasErrorEstimate() override { return false; }

  virtual void begin(const edm::EventSetup &) override;

  virtual void feed(const HitCollection &, const LocalTrackFit &) override {}

  virtual void saveDiagnostics(TDirectory *) override {}

  virtual std::vector<SingularMode> analyze() override;

  virtual unsigned int solve(const std::vector<AlignmentConstraint> &,
                             std::map<unsigned int, AlignmentResult> &result,
                             TDirectory *dir) override;

  virtual void end() override {}
};
