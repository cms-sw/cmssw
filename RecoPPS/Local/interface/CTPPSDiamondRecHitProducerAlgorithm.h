/****************************************************************************
*
* This is a part of PPS offline software.
* Authors:
*   Laurent Forthomme (laurent.forthomme@cern.ch)
*
****************************************************************************/

#ifndef RecoPPS_Local_CTPPSDiamondRecHitProducerAlgorithm
#define RecoPPS_Local_CTPPSDiamondRecHitProducerAlgorithm

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/Utils/interface/FormulaEvaluator.h"

#include "DataFormats/Common/interface/DetSetVector.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDiamondDetId.h"
#include "DataFormats/CTPPSDigi/interface/CTPPSDiamondDigi.h"
#include "DataFormats/CTPPSReco/interface/CTPPSDiamondRecHit.h"

#include "Geometry/VeryForwardRPTopology/interface/RPTopology.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSGeometry.h"

#include "CondFormats/PPSObjects/interface/PPSTimingCalibration.h"

class CTPPSDiamondRecHitProducerAlgorithm {
public:
  CTPPSDiamondRecHitProducerAlgorithm(const edm::ParameterSet& conf);

  void setCalibration(const PPSTimingCalibration&);
  void build(const CTPPSGeometry&, const edm::DetSetVector<CTPPSDiamondDigi>&, edm::DetSetVector<CTPPSDiamondRecHit>&);

private:
  static constexpr unsigned short MAX_CHANNEL = 20;
  /// Conversion constant between HPTDC time slice and absolute time (in ns)
  double ts_to_ns_;
  /// Switch on/off the timing calibration
  bool apply_calib_;
  PPSTimingCalibration calib_;
  std::unique_ptr<reco::FormulaEvaluator> calib_fct_;
};

#endif
