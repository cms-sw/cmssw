#ifndef RECOLOCALTRACKER_SISTRIPZEROSUPPRESSION_SISTRIPAPVRESTORER_H
#define RECOLOCALTRACKER_SISTRIPZEROSUPPRESSION_SISTRIPAPVRESTORER_H

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESWatcher.h"
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
  SiStripAPVRestorer(const edm::ParameterSet& conf, edm::ConsumesCollector);

public:
  virtual ~SiStripAPVRestorer(){};

  using digi_t = int16_t;
  using digivector_t = std::vector<digi_t>;
  using digimap_t = std::map<uint16_t, digi_t>;
  using medians_t = std::vector<std::pair<short, float>>;
  using baselinemap_t = std::map<uint16_t, digivector_t>;

  void init(const edm::EventSetup& es);

  uint16_t inspect(uint32_t detId, uint16_t firstAPV, const digivector_t& digis, const medians_t& vmedians);
  void restore(uint16_t firstAPV, digivector_t& digis);

  uint16_t inspectAndRestore(uint32_t detId,
                             uint16_t firstAPV,
                             const digivector_t& rawDigisPedSubtracted,
                             digivector_t& processedRawDigi,
                             const medians_t& vmedians);

  void loadMeanCMMap(const edm::Event&);

  const baselinemap_t& getBaselineMap() const { return baselineMap_; }
  const std::map<uint16_t, digimap_t>& getSmoothedPoints() const { return smoothedMaps_; }
  const std::vector<bool>& getAPVFlags() const { return apvFlagsBool_; }

private:
  using CMMap = std::map<uint32_t, std::vector<float>>;  //detId, Vector of MeanCM per detId
  constexpr static uint16_t nTotStripsPerAPV = 128;

  uint16_t nullInspect(uint16_t firstAPV, const digivector_t& digis);
  uint16_t abnormalBaselineInspect(uint16_t firstAPV, const digivector_t& digis);
  uint16_t baselineFollowerInspect(uint16_t firstAPV, const digivector_t& digis);
  uint16_t baselineAndSaturationInspect(uint16_t firstAPV, const digivector_t& digis);
  uint16_t forceRestoreInspect(uint16_t firstAPV, const digivector_t& digis);
  uint16_t hybridFormatInspect(uint16_t firstAPV, const digivector_t& digis);
  uint16_t hybridEmulationInspect(uint16_t firstAPV, const digivector_t& digis);

  void flatRestore(uint16_t apvN, uint16_t firstAPV, digivector_t& digis);
  bool checkBaseline(const std::vector<int16_t>& baseline) const;
  void baselineFollowerRestore(uint16_t apvN, uint16_t firstAPV, float median, digivector_t& digis);
  void derivativeFollowerRestore(uint16_t apvN, uint16_t firstAPV, digivector_t& digis);

  void baselineFollower(const digimap_t&, digivector_t& baseline, float median);
  bool flatRegionsFinder(const digivector_t& adcs, digimap_t& smoothedpoints, uint16_t apvN);

  void baselineCleaner(const digivector_t& adcs, digimap_t& smoothedpoints, uint16_t apvN);
  void cleaner_MonotonyChecker(digimap_t& smoothedpoints);
  void cleaner_HighSlopeChecker(digimap_t& smoothedpoints);
  void cleaner_LocalMinimumAdder(const digivector_t& adcs, digimap_t& smoothedpoints, uint16_t apvN);

  void createCMMapRealPed(const edm::DetSetVector<SiStripRawDigi>& input);
  void createCMMapCMstored(const edm::DetSetVector<SiStripProcessedRawDigi>& input);

private:  // members
  edm::ESGetToken<SiStripQuality, SiStripQualityRcd> qualityToken_;
  edm::ESGetToken<SiStripNoises, SiStripNoisesRcd> noiseToken_;
  edm::ESGetToken<SiStripPedestals, SiStripPedestalsRcd> pedestalToken_;
  const SiStripQuality* qualityHandle;
  const SiStripNoises* noiseHandle;
  const SiStripPedestals* pedestalHandle;
  edm::ESWatcher<SiStripQualityRcd> qualityWatcher_;
  edm::ESWatcher<SiStripNoisesRcd> noiseWatcher_;
  edm::ESWatcher<SiStripPedestalsRcd> pedestalWatcher_;

  // event state
  CMMap meanCMmap_;
  // state
  uint32_t detId_;
  std::vector<std::string> apvFlags_;
  std::vector<bool> apvFlagsBool_;
  std::vector<bool> apvFlagsBoolOverride_;
  std::vector<float> median_;
  std::vector<bool> badAPVs_;
  std::map<uint16_t, digimap_t> smoothedMaps_;
  baselinemap_t baselineMap_;

  //--------------------------------------------------
  // Configurable Parameters of Algorithm
  //--------------------------------------------------
  bool forceNoRestore_;
  std::string inspectAlgo_;
  std::string restoreAlgo_;
  bool useRealMeanCM_;
  int32_t meanCM_;
  uint32_t deltaCMThreshold_;     // for BaselineFollower inspect
  double fraction_;               // fraction of strips deviating from nominal baseline
  uint32_t deviation_;            // ADC value of deviation from nominal baseline
  double restoreThreshold_;       // for Null inspect (fraction of adc=0 channels)
  uint32_t nSaturatedStrip_;      // for BaselineAndSaturation inspect
  uint32_t nSigmaNoiseDerTh_;     // threshold for rejecting hits strips
  uint32_t consecThreshold_;      // minimum length of flat region
  uint32_t nSmooth_;              // for smoothing and local minimum determination (odd number)
  uint32_t distortionThreshold_;  // (max-min) of flat regions to trigger baseline follower
  bool applyBaselineCleaner_;
  uint32_t cleaningSequence_;
  int32_t slopeX_;
  int32_t slopeY_;
  uint32_t hitStripThreshold_;  // height above median when strip is definitely a hit
  uint32_t minStripsToFit_;     // minimum strips to try spline algo (otherwise default to median)
  bool applyBaselineRejection_;
  double filteredBaselineMax_;
  double filteredBaselineDerivativeSumSquare_;
  int gradient_threshold_;
  int last_gradient_;
  int size_window_;
  int width_cluster_;
};
#endif
