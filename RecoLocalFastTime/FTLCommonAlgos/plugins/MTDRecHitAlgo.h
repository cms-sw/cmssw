#include "RecoLocalFastTime/FTLCommonAlgos/interface/MTDRecHitAlgoBase.h"
#include "RecoLocalFastTime/Records/interface/MTDTimeCalibRecord.h"
#include "RecoLocalFastTime/FTLCommonAlgos/interface/MTDTimeCalib.h"

class MTDRecHitAlgo : public MTDRecHitAlgoBase {
public:
  /// Constructor
  MTDRecHitAlgo(const edm::ParameterSet& conf, edm::ConsumesCollector& sumes);

  /// Destructor
  ~MTDRecHitAlgo() override {}

  /// get event and eventsetup information
  void getEvent(const edm::Event&) final {}
  void getEventSetup(const edm::EventSetup&) final;

  /// make the rec hit
  FTLRecHit makeRecHit(const FTLUncalibratedRecHit& uRecHit, uint32_t& flags) const final;

  /// Fill parameter descriptions for validation
  static void fillPSetDescription(edm::ParameterSetDescription& desc);

private:
  double thresholdToKeep_, calibration_;
  const MTDTimeCalib* time_calib_;
  edm::ESGetToken<MTDTimeCalib, MTDTimeCalibRecord> tcToken_;
};
