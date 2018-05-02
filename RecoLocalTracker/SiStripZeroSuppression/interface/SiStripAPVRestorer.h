#ifndef RECOLOCALTRACKER_SISTRIPZEROSUPPRESSION_SISTRIPAPVRESTORER_H
#define RECOLOCALTRACKER_SISTRIPZEROSUPPRESSION_SISTRIPAPVRESTORER_H

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include "CondFormats/DataRecord/interface/SiStripPedestalsRcd.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripProcessedRawDigi.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripCommonModeNoiseSubtractor.h"

#include <vector>
#include <cstdint>

class SiStripAPVRestorer {
 friend class SiStripRawProcessingFactory;
protected:
  SiStripAPVRestorer(const edm::ParameterSet& conf);
public:
  virtual ~SiStripAPVRestorer() {};

  using digi_t = int16_t;
  using digivector_t = std::vector<digi_t>;
  using digimap_t = std::map<uint16_t, digi_t>;
  using medians_t = std::vector<std::pair<short,float>>;
  using baselinemap_t = std::map<uint16_t, digivector_t>;

  void init(const edm::EventSetup& es);

  uint16_t inspect(uint32_t detId, uint16_t firstAPV, const digivector_t& digis, const medians_t& vmedians);
  void restore(uint16_t firstAPV, digivector_t& digis);

  uint16_t InspectAndRestore(uint32_t detId, uint16_t firstAPV, const digivector_t& rawDigisPedSubtracted, digivector_t& processedRawDigi, const medians_t& vmedians);

  void LoadMeanCMMap(const edm::Event&);

  const baselinemap_t& GetBaselineMap() const { return BaselineMap_; }
  const std::map<uint16_t, digimap_t>& GetSmoothedPoints() const { return SmoothedMaps_; }
  const std::vector<bool>& GetAPVFlags() const { return apvFlagsBool_; }

private:
  using CMMap = std::map<uint32_t, std::vector<float>>;  //detId, Vector of MeanCM per detId

  uint16_t NullInspect(uint16_t firstAPV, const digivector_t& digis);
  uint16_t AbnormalBaselineInspect(uint16_t firstAPV, const digivector_t& digis);
  uint16_t BaselineFollowerInspect(uint16_t firstAPV, const digivector_t& digis);
  uint16_t BaselineAndSaturationInspect(uint16_t firstAPV, const digivector_t& digis);
  uint16_t ForceRestoreInspect(uint16_t firstAPV, const digivector_t& digis);
  uint16_t HybridFormatInspect(uint16_t firstAPV, const digivector_t& digis);
  uint16_t HybridEmulationInspect(uint16_t firstAPV, const digivector_t& digis);

  void FlatRestore(uint16_t APVn, uint16_t firstAPV, digivector_t& digis);
  bool CheckBaseline(const std::vector<int16_t> & baseline) const;
  void BaselineFollowerRestore(uint16_t APVn, uint16_t firstAPV, float median, digivector_t& digis);
  void DerivativeFollowerRestore(uint16_t APVn, uint16_t firstAPV, digivector_t& digis);

  void BaselineFollower(const digimap_t&, digivector_t& baseline, float median);
  bool FlatRegionsFinder(const digivector_t& adcs, digimap_t& smoothedpoints, uint16_t APVn);

  void BaselineCleaner(const digivector_t& adcs, digimap_t& smoothedpoints, uint16_t APVn);
  void Cleaner_MonotonyChecker(digimap_t& smoothedpoints);
  void Cleaner_HighSlopeChecker(digimap_t& smoothedpoints);
  void Cleaner_LocalMinimumAdder(const digivector_t& adcs, digimap_t& smoothedpoints, uint16_t APVn);

  void CreateCMMapRealPed(const edm::DetSetVector<SiStripRawDigi>& input);
  void CreateCMMapCMstored(const edm::DetSetVector<SiStripProcessedRawDigi>& input);

private: // members
  edm::ESHandle<SiStripQuality>   qualityHandle;
  edm::ESHandle<SiStripNoises>    noiseHandle;
  edm::ESHandle<SiStripPedestals> pedestalHandle;
  uint32_t quality_cache_id, noise_cache_id, pedestal_cache_id;

  // event state
  CMMap MeanCMmap_;
  // state
  uint32_t detId_;
  std::vector<std::string> apvFlags_;
  std::vector<bool> apvFlagsBool_;
  std::vector<bool> apvFlagsBoolOverride_;
  std::vector<float> median_;
  std::vector<bool> badAPVs_;
  std::map<uint16_t, digimap_t> SmoothedMaps_;
  baselinemap_t BaselineMap_;

  //--------------------------------------------------
  // Configurable Parameters of Algorithm
  //--------------------------------------------------
  bool ForceNoRestore_;
 // bool SelfSelectRestoreAlgo_;
  std::string InspectAlgo_;
  std::string RestoreAlgo_;
  bool     useRealMeanCM_;
  int32_t  MeanCM_;
  uint32_t DeltaCMThreshold_;          // for BaselineFollower inspect
  double   fraction_;                  // fraction of strips deviating from nominal baseline
  uint32_t deviation_;                 // ADC value of deviation from nominal baseline
  double   restoreThreshold_;          // for Null inspect (fraction of adc=0 channels)
  uint32_t nSaturatedStrip_;           // for BaselineAndSaturation inspect
  uint32_t nSigmaNoiseDerTh_;          // threshold for rejecting hits strips
  uint32_t consecThreshold_;           // minimum length of flat region
  uint32_t nSmooth_;                   // for smoothing and local minimum determination (odd number)
  uint32_t distortionThreshold_;       // (max-min) of flat regions to trigger baseline follower
  bool     ApplyBaselineCleaner_;
  uint32_t CleaningSequence_;
  int32_t  slopeX_;
  int32_t  slopeY_;
  uint32_t hitStripThreshold_;         // height above median when strip is definitely a hit
  uint32_t minStripsToFit_;            // minimum strips to try spline algo (otherwise default to median)
  bool    ApplyBaselineRejection_;
  double  filteredBaselineMax_;
  double  filteredBaselineDerivativeSumSquare_;
  int gradient_threshold_;
  int last_gradient_;
  int size_window_;
  int width_cluster_;
};
#endif
