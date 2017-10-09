// -*- C++ -*-
//
// Package:    HcalDbProducer
// Class:      HcalDbProducer
// 
/**\class HcalDbProducer HcalDbProducer.h CalibFormats/HcalDbProducer/interface/HcalDbProducer.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Fedor Ratnikov
//         Created:  Tue Aug  9 19:10:10 CDT 2005
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

class HcalDbService;
class HcalDbRecord;

#include "CondFormats/DataRecord/interface/HcalAllRcds.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"


class HcalDbProducer : public edm::ESProducer {
 public:
  HcalDbProducer( const edm::ParameterSet& );
  ~HcalDbProducer();
  
  std::shared_ptr<HcalDbService> produce( const HcalDbRecord& );

  std::shared_ptr<HcalChannelQuality> produceChannelQualityWithTopo( const HcalChannelQualityRcd&);

  // callbacks
  void pedestalsCallback (const HcalPedestalsRcd& fRecord);
  void pedestalWidthsCallback (const HcalPedestalWidthsRcd& fRecord);
  void gainsCallback (const HcalGainsRcd& fRecord);
  void gainWidthsCallback (const HcalGainWidthsRcd& fRecord);
  void QIEDataCallback (const HcalQIEDataRcd& fRecord);
  void QIETypesCallback (const HcalQIETypesRcd& fRecord);
  void channelQualityCallback (const HcalChannelQualityRcd& fRecord);
  void zsThresholdsCallback (const HcalZSThresholdsRcd& fRecord);
  void respCorrsCallback (const HcalRespCorrsRcd& fRecord);
  void L1triggerObjectsCallback (const HcalL1TriggerObjectsRcd& fRecord);
  void electronicsMapCallback (const HcalElectronicsMapRcd& fRecord);
  void frontEndMapCallback (const HcalFrontEndMapRcd& fRecord);
  void timeCorrsCallback (const HcalTimeCorrsRcd& fRecord);
  void LUTCorrsCallback (const HcalLUTCorrsRcd& fRecord);
  void PFCorrsCallback (const HcalPFCorrsRcd& fRecord);
  void lutMetadataCallback (const HcalLutMetadataRcd& fRecord);
  void SiPMParametersCallback (const HcalSiPMParametersRcd& fRecord);
  void SiPMCharacteristicsCallback (const HcalSiPMCharacteristicsRcd& fRecord);
  void TPChannelParametersCallback (const HcalTPChannelParametersRcd& fRecord);
  void TPParametersCallback (const HcalTPParametersRcd& fRecord);
  void MCParamsCallback (const HcalMCParamsRcd& fRecord);

private:
      // ----------member data ---------------------------
  std::shared_ptr<HcalDbService> mService;
  std::vector<std::string> mDumpRequest;
  std::ostream* mDumpStream;

  std::unique_ptr<HcalPedestals> mPedestals;
  std::unique_ptr<HcalPedestalWidths> mPedestalWidths;
  std::unique_ptr<HcalGains> mGains;
  std::unique_ptr<HcalGainWidths> mGainWidths;
  std::unique_ptr<HcalQIEData> mQIEData;
  std::unique_ptr<HcalQIETypes> mQIETypes;
  std::unique_ptr<HcalRespCorrs> mRespCorrs;
  std::unique_ptr<HcalLUTCorrs> mLUTCorrs;
  std::unique_ptr<HcalPFCorrs> mPFCorrs;
  std::unique_ptr<HcalTimeCorrs> mTimeCorrs;
  std::unique_ptr<HcalZSThresholds> mZSThresholds;
  std::unique_ptr<HcalL1TriggerObjects> mL1TriggerObjects;
  std::unique_ptr<HcalLutMetadata> mLutMetadata;
  std::unique_ptr<HcalSiPMParameters> mSiPMParameters;
  std::unique_ptr<HcalSiPMCharacteristics> mSiPMCharacteristics;
  std::unique_ptr<HcalTPChannelParameters> mTPChannelParameters;
  std::unique_ptr<HcalTPParameters> mTPParameters;
  std::unique_ptr<HcalMCParams> mMCParams;

};
