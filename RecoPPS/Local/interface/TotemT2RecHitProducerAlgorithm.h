/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors:
*   Laurent Forthomme (laurent.forthomme@cern.ch)
*
****************************************************************************/

#ifndef RecoPPS_Local_TotemT2RecHitProducerAlgorithm
#define RecoPPS_Local_TotemT2RecHitProducerAlgorithm

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/Utils/interface/FormulaEvaluator.h"

#include "DataFormats/Common/interface/DetSetVector.h"

#include "DataFormats/CTPPSDetId/interface/TotemT2DetId.h"
#include "DataFormats/TotemReco/interface/TotemT2Digi.h"
#include "DataFormats/TotemReco/interface/TotemT2RecHit.h"

#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSGeometry.h"

#include "CondFormats/PPSObjects/interface/PPSTimingCalibration.h"

class TotemT2RecHitProducerAlgorithm {
public:
  TotemT2RecHitProducerAlgorithm(const edm::ParameterSet& conf);

  void setCalibration(const PPSTimingCalibration&);
  void build(const CTPPSGeometry&, const edm::DetSetVector<TotemT2Digi>&, edm::DetSetVector<TotemT2RecHit>&);

private:
  /// Conversion constant between HPTDC time slice and absolute time (in ns)
  double ts_to_ns_;
  /// Switch on/off the timing calibration
  bool apply_calib_;
  PPSTimingCalibration calib_;
  std::unique_ptr<reco::FormulaEvaluator> calib_fct_;
};

#endif
