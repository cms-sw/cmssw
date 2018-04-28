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
  void initialize(const edm::EventSetup&);
  void initialize(const edm::EventSetup&, const edm::Event&);

  uint16_t SuppressHybridData(const edm::DetSet<SiStripDigi>& inDigis, edm::DetSet<SiStripDigi>& suppressedDigis, std::vector<int16_t>& rawDigis);
  uint16_t SuppressHybridData(uint32_t detId, uint16_t firstAPV, std::vector<int16_t>& processedRawDigis, edm::DetSet<SiStripDigi>& suppressedDigis);

  uint16_t SuppressVirginRawData(uint32_t detId, uint16_t firstAPV, std::vector<int16_t>& procRawDigis, edm::DetSet<SiStripDigi>& output);
  uint16_t SuppressVirginRawData(const edm::DetSet<SiStripRawDigi>& rawDigis, edm::DetSet<SiStripDigi>& output);

  uint16_t SuppressProcessedRawData(uint32_t detId, uint16_t firstAPV, std::vector<int16_t>& procRawDigis, edm::DetSet<SiStripDigi>& output);
  uint16_t SuppressProcessedRawData(const edm::DetSet<SiStripRawDigi>& rawDigis, edm::DetSet<SiStripDigi>& output);


  uint16_t ConvertVirginRawToHybrid(uint32_t detId, uint16_t firstAPV, std::vector<int16_t>& inDigis, edm::DetSet<SiStripDigi>& rawDigis);
  uint16_t ConvertVirginRawToHybrid(const edm::DetSet<SiStripRawDigi>& rawDigis, edm::DetSet<SiStripDigi>& suppressedDigis);

  void ConvertHybridDigiToRawDigiVector(uint32_t detId, const edm::DetSet<SiStripDigi>& inDigis, std::vector<int16_t>& rawDigis);

  inline std::vector<bool>& GetAPVFlags() { return restorer->GetAPVFlags(); }
  inline std::map<uint16_t, std::vector<int16_t>>& GetBaselineMap() { return restorer->GetBaselineMap(); }
  inline std::map<uint16_t, std::map<uint16_t,int16_t>>& GetSmoothedPoints() { return restorer->GetSmoothedPoints(); }
  inline const std::vector<std::pair<short,float>>& getAPVsCM() { return subtractorCMN->getAPVsCM(); }

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

