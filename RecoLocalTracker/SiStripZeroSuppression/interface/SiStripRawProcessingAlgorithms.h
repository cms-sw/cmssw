#ifndef RecoLocalTracker_SiStripZeroSuppression_SiStripRawProcessingAlgorithms_h
#define RecoLocalTracker_SiStripZeroSuppression_SiStripRawProcessingAlgorithms_h

#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripPedestalsSubtractor.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripCommonModeNoiseSubtractor.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripFedZeroSuppression.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripAPVRestorer.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"

#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"

class SiStripRawProcessingAlgorithms
{
  friend class SiStripRawProcessingFactory;

public:
  using digivector_t = SiStripAPVRestorer::digivector_t;

  void initialize(const edm::EventSetup&);
  void initialize(const edm::EventSetup&, const edm::Event&);

  uint16_t suppressHybridData(const edm::DetSet<SiStripDigi>& inDigis, edm::DetSet<SiStripDigi>& suppressedDigis, digivector_t& rawDigis);
  uint16_t suppressHybridData(uint32_t detId, uint16_t firstAPV, digivector_t& processedRawDigis, edm::DetSet<SiStripDigi>& suppressedDigis);

  uint16_t suppressVirginRawData(uint32_t detId, uint16_t firstAPV, digivector_t& procRawDigis, edm::DetSet<SiStripDigi>& output);
  uint16_t suppressVirginRawData(const edm::DetSet<SiStripRawDigi>& rawDigis, edm::DetSet<SiStripDigi>& output);

  uint16_t suppressProcessedRawData(uint32_t detId, uint16_t firstAPV, digivector_t& procRawDigis, edm::DetSet<SiStripDigi>& output);
  uint16_t suppressProcessedRawData(const edm::DetSet<SiStripRawDigi>& rawDigis, edm::DetSet<SiStripDigi>& output);


  uint16_t convertVirginRawToHybrid(uint32_t detId, uint16_t firstAPV, digivector_t& inDigis, edm::DetSet<SiStripDigi>& rawDigis);
  uint16_t convertVirginRawToHybrid(const edm::DetSet<SiStripRawDigi>& rawDigis, edm::DetSet<SiStripDigi>& suppressedDigis);

  void convertHybridDigiToRawDigiVector(const edm::DetSet<SiStripDigi>& inDigis, digivector_t& rawDigis);

  inline const std::vector<bool>& getAPVFlags() const { return restorer->getAPVFlags(); }
  inline const SiStripAPVRestorer::baselinemap_t& getBaselineMap() const { return restorer->getBaselineMap(); }
  inline const std::map<uint16_t, SiStripAPVRestorer::digimap_t>& getSmoothedPoints() const { return restorer->getSmoothedPoints(); }
  inline const SiStripAPVRestorer::medians_t& getAPVsCM() const { return subtractorCMN->getAPVsCM(); }

  const std::unique_ptr<SiStripPedestalsSubtractor> subtractorPed;
  const std::unique_ptr<SiStripCommonModeNoiseSubtractor> subtractorCMN;
  const std::unique_ptr<SiStripFedZeroSuppression> suppressor;
  const std::unique_ptr<SiStripAPVRestorer> restorer;

 private:
  const bool doAPVRestore;
  const bool useCMMeanMap;

  const TrackerGeometry* trGeo;

  SiStripRawProcessingAlgorithms(std::unique_ptr<SiStripPedestalsSubtractor> ped,
				 std::unique_ptr<SiStripCommonModeNoiseSubtractor> cmn,
				 std::unique_ptr<SiStripFedZeroSuppression> zs,
                                 std::unique_ptr<SiStripAPVRestorer> res,
				 bool doAPVRest,
				 bool useCMMap);
};
#endif
