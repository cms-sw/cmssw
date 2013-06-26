#ifndef RecoLocalTracker_SiStripZeroSuppression_SiStripRawProcessingAlgorithms_h
#define RecoLocalTracker_SiStripZeroSuppression_SiStripRawProcessingAlgorithms_h

#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripPedestalsSubtractor.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripCommonModeNoiseSubtractor.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripFedZeroSuppression.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripAPVRestorer.h"

#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"

class SiStripRawProcessingAlgorithms {
  friend class SiStripRawProcessingFactory;

 public: 
  void initialize(const edm::EventSetup&);
  void initialize(const edm::EventSetup&, const edm::Event&);
  int16_t SuppressVirginRawData(const uint32_t&, const uint16_t&, std::vector<int16_t>&, edm::DetSet<SiStripDigi>&);
  int16_t SuppressVirginRawData(const edm::DetSet<SiStripRawDigi>&, edm::DetSet<SiStripDigi>& );
  
  int16_t SuppressProcessedRawData(const uint32_t&, const uint16_t&, std::vector<int16_t>&, edm::DetSet<SiStripDigi>&);
  int16_t SuppressProcessedRawData(const edm::DetSet<SiStripRawDigi>&, edm::DetSet<SiStripDigi>&  );

  inline std::vector<bool>& GetAPVFlags(){return restorer->GetAPVFlags();}
  inline std::map<uint16_t, std::vector < int16_t> >& GetBaselineMap(){return restorer->GetBaselineMap();}
  inline std::map< uint16_t, std::map< uint16_t, int16_t> >& GetSmoothedPoints(){return restorer->GetSmoothedPoints();}
  inline const std::vector< std::pair<short,float> >& getAPVsCM(){return subtractorCMN->getAPVsCM();}

  const std::auto_ptr<SiStripPedestalsSubtractor> subtractorPed;
  const std::auto_ptr<SiStripCommonModeNoiseSubtractor> subtractorCMN;
  const std::auto_ptr<SiStripFedZeroSuppression> suppressor;
  const std::auto_ptr<SiStripAPVRestorer> restorer;

 private:
  const bool doAPVRestore;
  const bool useCMMeanMap;

  SiStripRawProcessingAlgorithms(std::auto_ptr<SiStripPedestalsSubtractor> ped,
				 std::auto_ptr<SiStripCommonModeNoiseSubtractor> cmn,
				 std::auto_ptr<SiStripFedZeroSuppression> zs,
                                 std::auto_ptr<SiStripAPVRestorer> res,
				 bool doAPVRest,
				 bool useCMMap); 
   
};
#endif

