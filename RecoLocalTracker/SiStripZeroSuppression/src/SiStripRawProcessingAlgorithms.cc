#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripRawProcessingAlgorithms.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripRawProcessingFactory.h"
#include <memory>

SiStripRawProcessingAlgorithms::SiStripRawProcessingAlgorithms(edm::ConsumesCollector iC,
                                                               std::unique_ptr<SiStripPedestalsSubtractor> ped,
                                                               std::unique_ptr<SiStripCommonModeNoiseSubtractor> cmn,
                                                               std::unique_ptr<SiStripFedZeroSuppression> zs,
                                                               std::unique_ptr<SiStripAPVRestorer> res,
                                                               bool doAPVRest,
                                                               bool useCMMap)
    : subtractorPed(std::move(ped)),
      subtractorCMN(std::move(cmn)),
      suppressor(std::move(zs)),
      restorer(std::move(res)),
      doAPVRestore(doAPVRest),
      useCMMeanMap(useCMMap),
      tkGeomToken_(iC.esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>()) {}

void SiStripRawProcessingAlgorithms::initialize(const edm::EventSetup& es) {
  subtractorPed->init(es);
  subtractorCMN->init(es);
  suppressor->init(es);
  if (restorer.get())
    restorer->init(es);

  trGeo = &es.getData(tkGeomToken_);
}

void SiStripRawProcessingAlgorithms::initialize(const edm::EventSetup& es, const edm::Event& e) {
  initialize(es);
  if (restorer.get() && doAPVRestore && useCMMeanMap)
    restorer->loadMeanCMMap(e);
}

/**
 * Zero-suppress "hybrid" raw data
 *
 * Subtracts common-mode noise, and inspects the digis then.
 * If flagged by the APV inspector, the zero-suppression is performed as usual.
 * Otherwise, the positive inputs are copied.
 *
 * @param hybridDigis input ADCs in ZS format (regular ZS or "hybrid", i.e. processed as x->(x+1024-ped)/2)
 * @param suppressedDigis zero-suppressed digis
 * @param firstAPV (optional) number of the first APV for which digis are should be handled (otherwise all present)
 *
 * @return number of restored APVs
 */
uint16_t SiStripRawProcessingAlgorithms::suppressHybridData(const uint16_t maxNStrips,
                                                            const edm::DetSet<SiStripDigi>& hybridDigis,
                                                            edm::DetSet<SiStripDigi>& suppressedDigis,
                                                            uint16_t firstAPV) {
  uint16_t nAPVFlagged = 0;  // Count of flagged APVs
  auto currentDigi = hybridDigis.begin();
  const auto endDigi = hybridDigis.end();
  auto currentAPV = firstAPV;

  // Loop through the APVs in the range
  while (currentDigi != endDigi) {
    // Determine the range of digis belonging to the current APV
    const auto nextAPVBoundary = SiStripDigi((currentAPV + 1) * 128, 0);

    // Reject any APV larger than the max possible
    if (nextAPVBoundary.strip() > maxNStrips) {
      edm::LogError("SiStripRawProcessingAlgorithms")
          << "In DetId " << suppressedDigis.id << " encountered APV boundary with strip number "
          << nextAPVBoundary.strip() << ", which exceeds the maximum allowed value for this module (" << maxNStrips
          << "). Exiting loop.";
      break;
    }

    const auto nextAPVDigi = std::lower_bound(currentDigi, endDigi, nextAPVBoundary);
    const auto nDigisInAPV = std::distance(currentDigi, nextAPVDigi);

    // Handle based on the number of digis in the current APV
    if (nDigisInAPV > 64) {
      // Process hybrid data for noisy APV
      digivector_t workDigis(128, -1024);

      // Populate `workDigis` with values from `currentDigi`
      for (auto it = currentDigi; it != nextAPVDigi; ++it) {
        workDigis[it->strip() - 128 * currentAPV] = it->adc() * 2 - 1024;
      }

      // Perform pedestal subtraction
      digivector_t workDigisPedSubtracted(workDigis);
      subtractorCMN->subtract(hybridDigis.id, currentAPV, workDigis);

      // Inspect and restore digis
      const auto apvFlagged = restorer->inspectAndRestore(
          hybridDigis.id, currentAPV, workDigisPedSubtracted, workDigis, subtractorCMN->getAPVsCM());
      nAPVFlagged += apvFlagged;

      // Process based on the APV flag
      if (getAPVFlags()[currentAPV]) {
        // Suppress flagged APVs
        suppressor->suppress(workDigis, currentAPV, suppressedDigis);
      } else {
        // Handle bad APV: not flagged but exceeds threshold
        for (uint16_t i = 0; i < 128; ++i) {
          const auto digi = workDigisPedSubtracted[i];
          if (digi > 0) {
            suppressedDigis.push_back(SiStripDigi(currentAPV * 128 + i, suppressor->truncate(digi)));
          }
        }
      }
    } else {
      // Already zero-suppressed: copy and truncate
      std::transform(currentDigi, nextAPVDigi, std::back_inserter(suppressedDigis), [this](const SiStripDigi& inDigi) {
        return SiStripDigi(inDigi.strip(), suppressor->truncate(inDigi.adc()));
      });
    }

    // Move to the next APV
    currentDigi = nextAPVDigi;
    ++currentAPV;
  }

  return nAPVFlagged;
}

/**
 * Zero-suppress virgin raw data.
 *
 * Subtracts pedestals and common-mode noise, and (optionally, if doAPVRestore)
 * re-evaluates and subtracts the baseline.
 *
 * @param id module DetId
 * @param firstAPV index of the first APV to consider
 * @param procRawDigis input (virgin raw) ADCs. Output: the ADCs after all subtractions, but before zero-suppression
 * @param output zero-suppressed digis
 * @return number of restored APVs
 */
uint16_t SiStripRawProcessingAlgorithms::suppressVirginRawData(uint32_t id,
                                                               uint16_t firstAPV,
                                                               digivector_t& procRawDigis,
                                                               edm::DetSet<SiStripDigi>& output) {
  subtractorPed->subtract(id, firstAPV * 128, procRawDigis);
  return suppressProcessedRawData(id, firstAPV, procRawDigis, output);
}

/**
 * Zero-suppress virgin raw data.
 *
 * Subtracts pedestals and common-mode noise, and (optionally, if doAPVRestore)
 * re-evaluates and subtracts the baseline.
 *
 * @param rawDigis input (virgin) raw digis
 * @param output zero-suppressed digis
 * @return number of restored APVs
 */
uint16_t SiStripRawProcessingAlgorithms::suppressVirginRawData(const edm::DetSet<SiStripRawDigi>& rawDigis,
                                                               edm::DetSet<SiStripDigi>& output) {
  digivector_t rawdigis;
  rawdigis.reserve(rawDigis.size());
  std::transform(std::begin(rawDigis), std::end(rawDigis), std::back_inserter(rawdigis), [](SiStripRawDigi digi) {
    return digi.adc();
  });
  return suppressVirginRawData(rawDigis.id, 0, rawdigis, output);
}

/**
 * Zero-suppress processed (pedestals-subtracted) raw data.
 *
 * Subtracts common-mode noise and (optionally, if doAPVRestore)
 * re-evaluates and subtracts the baseline.
 *
 * @param id module DetId
 * @param firstAPV index of the first APV to consider
 * @param procRawDigis input (processed raw) ADCs. Output: the ADCs after all subtractions, but before zero-suppression
 * @param output zero-suppressed digis
 * @return number of restored APVs
 */
uint16_t SiStripRawProcessingAlgorithms::suppressProcessedRawData(uint32_t id,
                                                                  uint16_t firstAPV,
                                                                  digivector_t& procRawDigis,
                                                                  edm::DetSet<SiStripDigi>& output) {
  digivector_t procRawDigisPedSubtracted;

  int16_t nAPVFlagged = 0;
  if (doAPVRestore)
    procRawDigisPedSubtracted.assign(procRawDigis.begin(), procRawDigis.end());
  subtractorCMN->subtract(id, firstAPV, procRawDigis);
  if (doAPVRestore)
    nAPVFlagged =
        restorer->inspectAndRestore(id, firstAPV, procRawDigisPedSubtracted, procRawDigis, subtractorCMN->getAPVsCM());
  suppressor->suppress(procRawDigis, firstAPV, output);
  return nAPVFlagged;
}

/**
 * Zero-suppress processed (pedestals-subtracted) raw data.
 *
 * Subtracts common-mode noise and (optionally, if doAPVRestore)
 * re-evaluates and subtracts the baseline.
 *
 * @param rawDigis input (processed) raw digis
 * @param output zero-suppressed digis
 * @return number of restored APVs
 */
uint16_t SiStripRawProcessingAlgorithms::suppressProcessedRawData(const edm::DetSet<SiStripRawDigi>& rawDigis,
                                                                  edm::DetSet<SiStripDigi>& output) {
  digivector_t rawdigis;
  rawdigis.reserve(rawDigis.size());
  std::transform(std::begin(rawDigis), std::end(rawDigis), std::back_inserter(rawdigis), [](SiStripRawDigi digi) {
    return digi.adc();
  });
  return suppressProcessedRawData(rawDigis.id, 0, rawdigis, output);
}

/**
 * Zero-suppress virgin raw data in "hybrid" mode
 *
 * Subtracts pedestals (in 11bit mode, x->(x+1024-ped)/2) and common-mode noise, and inspects the digis then.
 * If not flagged by the hybrid APV inspector, the zero-suppression is performed as usual
 * (evaluation and subtraction of the baseline, truncation).
 * Otherwise, the pedestal-subtracted digis (as above) are saved in one 128-strip cluster.
 * Note: the APV restorer is used, it must be configured with APVInspectMode='HybridEmulation' if this method is called.
 *
 * @param id module DetId
 * @param firstAPV index of the first APV considered
 * @param procRawDigis input (virgin raw) ADCs. Output: the ADCs after all subtractions, but before zero-suppression
 * @param output zero-suppressed digis (or pedestal-subtracted digis, see above)
 * @return number of restored APVs
 */
uint16_t SiStripRawProcessingAlgorithms::convertVirginRawToHybrid(uint32_t id,
                                                                  uint16_t firstAPV,
                                                                  digivector_t& procRawDigis,
                                                                  edm::DetSet<SiStripDigi>& output) {
  digivector_t procRawDigisPedSubtracted;

  for (auto& digi : procRawDigis) {
    digi += 1024;
  }  // adding one MSB

  subtractorPed->subtract(id, firstAPV * 128, procRawDigis);  // all strips are pedestals subtracted

  for (auto& digi : procRawDigis) {
    digi /= 2;
  }

  procRawDigisPedSubtracted.assign(procRawDigis.begin(), procRawDigis.end());

  subtractorCMN->subtract(id, firstAPV, procRawDigis);

  const auto nAPVFlagged = restorer->inspect(id, firstAPV, procRawDigis, subtractorCMN->getAPVsCM());

  for (auto& digi : procRawDigis) {
    digi *= 2;
  }

  const std::vector<bool>& apvf = getAPVFlags();
  const std::size_t nAPVs = procRawDigis.size() / 128;
  for (uint16_t iAPV = firstAPV; iAPV < nAPVs + firstAPV; ++iAPV) {
    if (apvf[iAPV]) {
      //GB 23/6/08: truncation should be done at the very beginning
      for (uint16_t i = 0; i < 128; ++i) {
        const int16_t digi = procRawDigisPedSubtracted[128 * (iAPV - firstAPV) + i];
        output.push_back(SiStripDigi(128 * iAPV + i, (digi < 0 ? 0 : suppressor->truncate(digi))));
      }
    } else {
      const auto firstDigiIt = std::begin(procRawDigis) + 128 * (iAPV - firstAPV);
      std::vector<int16_t> singleAPVdigi(firstDigiIt, firstDigiIt + 128);
      suppressor->suppress(singleAPVdigi, iAPV, output);
    }
  }

  return nAPVFlagged;
}

/**
 * Zero-suppress virgin raw data in "hybrid" mode
 *
 * Subtracts pedestals (in 11bit mode, x->(x+1024-ped)/2) and common-mode noise, and inspects the digis then.
 * If flagged by the hybrid APV inspector, the zero-suppression is performed as usual
 * (evaluation and subtraction of the baseline, truncation).
 * Otherwise, the pedestal-subtracted digis are saved in one 128-strip cluster.
 *
 * @param rawDigis input (virgin) raw digis
 * @param output zero-suppressed digis (or pedestal-subtracted digis, see above)
 * @return number of restored APVs
 */
uint16_t SiStripRawProcessingAlgorithms::convertVirginRawToHybrid(const edm::DetSet<SiStripRawDigi>& rawDigis,
                                                                  edm::DetSet<SiStripDigi>& suppressedDigis) {
  digivector_t rawdigis;
  rawdigis.reserve(rawDigis.size());
  std::transform(std::begin(rawDigis), std::end(rawDigis), std::back_inserter(rawdigis), [](SiStripRawDigi digi) {
    return digi.adc();
  });
  return convertVirginRawToHybrid(rawDigis.id, 0, rawdigis, suppressedDigis);
}
