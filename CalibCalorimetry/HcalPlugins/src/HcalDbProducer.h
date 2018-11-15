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
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESProductHost.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Utilities/interface/ReusableObjectHolder.h"

#include "CondFormats/DataRecord/interface/HcalAllRcds.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"

class HcalDbProducer : public edm::ESProducer {
public:
  HcalDbProducer( const edm::ParameterSet& );
  ~HcalDbProducer() override;

  std::shared_ptr<HcalDbService> produce( const HcalDbRecord& );

  std::unique_ptr<HcalPedestals> producePedestalsWithTopo(const HcalPedestalsRcd&);
  std::unique_ptr<HcalPedestalWidths> producePedestalWidthsWithTopo(const HcalPedestalWidthsRcd&);
  std::unique_ptr<HcalPedestals> produceEffectivePedestalsWithTopo(const HcalPedestalsRcd&);
  std::unique_ptr<HcalPedestalWidths> produceEffectivePedestalWidthsWithTopo(const HcalPedestalWidthsRcd&);
  std::unique_ptr<HcalGains> produceGainsWithTopo(const HcalGainsRcd&);
  std::unique_ptr<HcalGainWidths> produceGainWidthsWithTopo(const HcalGainWidthsRcd&);
  std::unique_ptr<HcalQIEData> produceQIEDataWithTopo(const HcalQIEDataRcd&);
  std::unique_ptr<HcalQIETypes> produceQIETypesWithTopo(const HcalQIETypesRcd&);
  std::unique_ptr<HcalChannelQuality> produceChannelQualityWithTopo( const HcalChannelQualityRcd&);
  std::unique_ptr<HcalZSThresholds> produceZSThresholdsWithTopo(const HcalZSThresholdsRcd&);
  std::unique_ptr<HcalRespCorrs> produceRespCorrsWithTopo(const HcalRespCorrsRcd&);
  std::unique_ptr<HcalL1TriggerObjects> produceL1triggerObjectsWithTopo(const HcalL1TriggerObjectsRcd&);
  std::unique_ptr<HcalTimeCorrs> produceTimeCorrsWithTopo(const HcalTimeCorrsRcd&);
  std::unique_ptr<HcalLUTCorrs> produceLUTCorrsWithTopo(const HcalLUTCorrsRcd&);
  std::unique_ptr<HcalPFCorrs> producePFCorrsWithTopo(const HcalPFCorrsRcd&);
  std::unique_ptr<HcalLutMetadata> produceLUTMetadataWithTopo(const HcalLutMetadataRcd&);
  std::unique_ptr<HcalSiPMParameters> produceSiPMParametersWithTopo(const HcalSiPMParametersRcd&);
  std::unique_ptr<HcalTPChannelParameters> produceTPChannelParametersWithTopo(const HcalTPChannelParametersRcd&);
  std::unique_ptr<HcalMCParams> produceMCParamsWithTopo(const HcalMCParamsRcd&);
  std::unique_ptr<HcalRecoParams> produceRecoParamsWithTopo(const HcalRecoParamsRcd&);

  void setupPedestals(const HcalPedestalsRcd&, HcalDbService&);
  void setupPedestalWidths(const HcalPedestalWidthsRcd&, HcalDbService&);
  void setupEffectivePedestals(const HcalPedestalsRcd&, HcalDbService&);
  void setupEffectivePedestalWidths(const HcalPedestalWidthsRcd&, HcalDbService&);

private:

  using HostType = edm::ESProductHost<HcalDbService,
                                      HcalPedestalsRcd,
                                      HcalPedestalWidthsRcd,
                                      HcalGainsRcd,
                                      HcalGainWidthsRcd,
                                      HcalQIEDataRcd,
                                      HcalQIETypesRcd,
                                      HcalChannelQualityRcd,
                                      HcalZSThresholdsRcd,
                                      HcalRespCorrsRcd,
                                      HcalL1TriggerObjectsRcd,
                                      HcalTimeCorrsRcd,
                                      HcalLUTCorrsRcd,
                                      HcalPFCorrsRcd,
                                      HcalLutMetadataRcd,
                                      HcalSiPMParametersRcd,
                                      HcalTPChannelParametersRcd,
                                      HcalMCParamsRcd,
                                      HcalRecoParamsRcd,
                                      HcalElectronicsMapRcd,
                                      HcalFrontEndMapRcd,
                                      HcalSiPMCharacteristicsRcd,
                                      HcalTPParametersRcd>;

  template <typename ProductType, typename RecordType>
  static std::unique_ptr<ProductType> produceWithTopology(RecordType const& record) {

    edm::ESTransientHandle<ProductType> item;
    record.get(item);

    auto productWithTopology = std::make_unique<ProductType>(*item);

    edm::ESHandle<HcalTopology> htopo;
    record. template getRecord<HcalRecNumberingRecord>().get(htopo);
    const HcalTopology* topo=&(*htopo);
    productWithTopology->setTopo(topo);

    return productWithTopology;
  }

  template <typename ProductType, typename RecordType>
  void setupHcalDbService(HostType& host,
                          const HcalDbRecord& record,
                          const char* label,
                          const char* dumpName,
                          const char* dumpHeader) {

    host.ifRecordChanges<RecordType>(record,
                                     [this, &host, label, dumpName, dumpHeader](auto const& rec) {
      edm::ESHandle<ProductType> item;
      rec.get(label, item);
      host.setData(item.product());

      if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string (dumpName)) != mDumpRequest.end()) {
        *mDumpStream << dumpHeader << std::endl;
        HcalDbASCIIIO::dumpObject(*mDumpStream, *item);
      }
    });
  }

  // ----------member data ---------------------------
  edm::ReusableObjectHolder<HostType> holder_;

  std::vector<std::string> mDumpRequest;
  std::ostream* mDumpStream;
};
