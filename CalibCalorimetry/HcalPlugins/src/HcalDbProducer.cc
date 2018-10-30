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
#include <iostream>
#include <fstream>

#include "CondFormats/HcalObjects/interface/AllObjects.h"

#include "HcalDbProducer.h"

HcalDbProducer::HcalDbProducer( const edm::ParameterSet& fConfig)
  : ESProducer(),
    mDumpRequest (),
    mDumpStream(nullptr)
{
  setWhatProduced (this);

  setWhatProduced(this, &HcalDbProducer::producePedestalsWithTopo, edm::es::Label("withTopo"));
  setWhatProduced(this, &HcalDbProducer::producePedestalWidthsWithTopo, edm::es::Label("withTopo"));
  setWhatProduced(this, &HcalDbProducer::produceEffectivePedestalsWithTopo, edm::es::Label("withTopoEff"));
  setWhatProduced(this, &HcalDbProducer::produceEffectivePedestalWidthsWithTopo, edm::es::Label("withTopoEff"));
  setWhatProduced(this, &HcalDbProducer::produceGainsWithTopo, edm::es::Label("withTopo"));
  setWhatProduced(this, &HcalDbProducer::produceGainWidthsWithTopo, edm::es::Label("withTopo"));
  setWhatProduced(this, &HcalDbProducer::produceQIEDataWithTopo, edm::es::Label("withTopo"));
  setWhatProduced(this, &HcalDbProducer::produceQIETypesWithTopo, edm::es::Label("withTopo"));
  setWhatProduced(this, &HcalDbProducer::produceChannelQualityWithTopo, edm::es::Label("withTopo"));
  setWhatProduced(this, &HcalDbProducer::produceZSThresholdsWithTopo, edm::es::Label("withTopo"));
  setWhatProduced(this, &HcalDbProducer::produceRespCorrsWithTopo, edm::es::Label("withTopo"));
  setWhatProduced(this, &HcalDbProducer::produceL1triggerObjectsWithTopo, edm::es::Label("withTopo"));
  setWhatProduced(this, &HcalDbProducer::produceTimeCorrsWithTopo, edm::es::Label("withTopo"));
  setWhatProduced(this, &HcalDbProducer::produceLUTCorrsWithTopo, edm::es::Label("withTopo"));
  setWhatProduced(this, &HcalDbProducer::producePFCorrsWithTopo, edm::es::Label("withTopo"));
  setWhatProduced(this, &HcalDbProducer::produceLUTMetadataWithTopo, edm::es::Label("withTopo"));
  setWhatProduced(this, &HcalDbProducer::produceSiPMParametersWithTopo, edm::es::Label("withTopo"));
  setWhatProduced(this, &HcalDbProducer::produceTPChannelParametersWithTopo, edm::es::Label("withTopo"));
  setWhatProduced(this, &HcalDbProducer::produceMCParamsWithTopo, edm::es::Label("withTopo"));
  setWhatProduced(this, &HcalDbProducer::produceRecoParamsWithTopo, edm::es::Label("withTopo"));

  mDumpRequest = fConfig.getUntrackedParameter <std::vector <std::string> > ("dump", std::vector<std::string>());
  if (!mDumpRequest.empty()) {
    std::string otputFile = fConfig.getUntrackedParameter <std::string> ("file", "");
    mDumpStream = otputFile.empty () ? &std::cout : new std::ofstream (otputFile.c_str());
  }
}

HcalDbProducer::~HcalDbProducer() {

  if (mDumpStream != &std::cout) delete mDumpStream;
}

// ------------ method called to produce the data  ------------
std::shared_ptr<HcalDbService> HcalDbProducer::produce(const HcalDbRecord& record) {

  auto host = holder_.makeOrGet([]() {
    return new HostType;
  });

  bool pedestalWidthsChanged = false;
  host->ifRecordChanges<HcalPedestalWidthsRcd>(record,
                                               [this,h=host.get(),&pedestalWidthsChanged](auto const& rec) {
    setupEffectivePedestalWidths(rec, *h);
    pedestalWidthsChanged = true;
  });


  bool pedestalsChanged = false;
  host->ifRecordChanges<HcalPedestalsRcd>(record,
                                          [this,h=host.get(),&pedestalsChanged](auto const& rec) {
    setupEffectivePedestals(rec, *h);
    pedestalsChanged = true;
  });

  setupHcalDbService<HcalRecoParams, HcalRecoParamsRcd>(
    *host, record, "withTopo", "RecoParams", "New HCAL RecoParams set");

  setupHcalDbService<HcalMCParams, HcalMCParamsRcd>(
    *host, record, "withTopo", "MCParams", "New HCAL MCParams set");

  setupHcalDbService<HcalLutMetadata, HcalLutMetadataRcd>(
    *host, record, "withTopo", "LutMetadata", "New HCAL LUT Metadata set");

  setupHcalDbService<HcalTPParameters, HcalTPParametersRcd>(
    *host, record, "", "TPParameters", "New HCAL TPParameters set");

  setupHcalDbService<HcalTPChannelParameters, HcalTPChannelParametersRcd>(
    *host, record, "withTopo", "TPChannelParameters", "New HCAL TPChannelParameters set");

  setupHcalDbService<HcalSiPMCharacteristics, HcalSiPMCharacteristicsRcd>(
    *host, record, "", "SiPMCharacteristics", "New HCAL SiPMCharacteristics set");

  setupHcalDbService<HcalSiPMParameters, HcalSiPMParametersRcd>(
    *host, record, "withTopo", "SiPMParameters", "New HCAL SiPMParameters set");

  setupHcalDbService<HcalFrontEndMap, HcalFrontEndMapRcd>(
    *host, record, "", "FrontEndMap", "New HCAL FrontEnd Map set");

  setupHcalDbService<HcalElectronicsMap, HcalElectronicsMapRcd>(
    *host, record, "", "ElectronicsMap", "New HCAL Electronics Map set");

  setupHcalDbService<HcalL1TriggerObjects, HcalL1TriggerObjectsRcd>(
    *host, record, "withTopo", "L1TriggerObjects", "New HCAL L1TriggerObjects set");

  setupHcalDbService<HcalZSThresholds, HcalZSThresholdsRcd>(
    *host, record, "withTopo", "ZSThresholds", "New HCAL ZSThresholds set");

  setupHcalDbService<HcalChannelQuality, HcalChannelQualityRcd>(
    *host, record, "withTopo", "ChannelQuality", "New HCAL ChannelQuality set");

  setupHcalDbService<HcalGainWidths, HcalGainWidthsRcd>(
    *host, record, "withTopo", "GainWidths", "New HCAL GainWidths set");

  setupHcalDbService<HcalQIETypes, HcalQIETypesRcd>(
    *host, record, "withTopo", "QIETypes", "New HCAL QIETypes set");

  setupHcalDbService<HcalQIEData, HcalQIEDataRcd>(
    *host, record, "withTopo", "QIEData", "New HCAL QIEData set");

  setupHcalDbService<HcalTimeCorrs, HcalTimeCorrsRcd>(
    *host, record, "withTopo", "TimeCorrs", "New HCAL TimeCorrs set");

  setupHcalDbService<HcalPFCorrs, HcalPFCorrsRcd>(
    *host, record, "withTopo", "PFCorrs", "New HCAL PFCorrs set");

  setupHcalDbService<HcalLUTCorrs, HcalLUTCorrsRcd>(
    *host, record, "withTopo", "LUTCorrs", "New HCAL LUTCorrs set");

  setupHcalDbService<HcalGains, HcalGainsRcd>(
    *host, record, "withTopo", "Gains", "New HCAL Gains set");

  setupHcalDbService<HcalRespCorrs, HcalRespCorrsRcd>(
    *host, record, "withTopo", "RespCorrs", "New HCAL RespCorrs set");

  if (pedestalWidthsChanged) {
    HcalPedestalWidthsRcd const& rec = record.getRecord<HcalPedestalWidthsRcd>();
    setupPedestalWidths(rec, *host);
  }

  if (pedestalsChanged) {
    HcalPedestalsRcd const& rec = record.getRecord<HcalPedestalsRcd>();
    setupPedestals(rec, *host);
  }

  return host;
}

std::unique_ptr<HcalPedestals>
HcalDbProducer::producePedestalsWithTopo(const HcalPedestalsRcd& record) {
  return produceWithTopology<HcalPedestals>(record);
}

std::unique_ptr<HcalPedestalWidths>
HcalDbProducer::producePedestalWidthsWithTopo(const HcalPedestalWidthsRcd& record) {
  return produceWithTopology<HcalPedestalWidths>(record);
}

std::unique_ptr<HcalPedestals>
HcalDbProducer::produceEffectivePedestalsWithTopo(const HcalPedestalsRcd& record) {

  edm::ESTransientHandle<HcalPedestals> item;
  record.get("effective", item);

  auto productWithTopology = std::make_unique<HcalPedestals>(*item);

  edm::ESHandle<HcalTopology> htopo;
  record.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  productWithTopology->setTopo(topo);

  return productWithTopology;
}

std::unique_ptr<HcalPedestalWidths>
HcalDbProducer::produceEffectivePedestalWidthsWithTopo(const HcalPedestalWidthsRcd& record) {

  edm::ESTransientHandle<HcalPedestalWidths> item;
  record.get("effective", item);

  auto productWithTopology = std::make_unique<HcalPedestalWidths>(*item);

  edm::ESHandle<HcalTopology> htopo;
  record. template getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  productWithTopology->setTopo(topo);

  return productWithTopology;
}

std::unique_ptr<HcalGains>
HcalDbProducer::produceGainsWithTopo(const HcalGainsRcd& record) {
  return produceWithTopology<HcalGains>(record);
}

std::unique_ptr<HcalGainWidths>
HcalDbProducer::produceGainWidthsWithTopo(const HcalGainWidthsRcd& record) {
  return produceWithTopology<HcalGainWidths>(record);
}

std::unique_ptr<HcalQIEData>
HcalDbProducer::produceQIEDataWithTopo(const HcalQIEDataRcd& record) {
  return produceWithTopology<HcalQIEData>(record);
}

std::unique_ptr<HcalQIETypes>
HcalDbProducer::produceQIETypesWithTopo(const HcalQIETypesRcd& record) {
  return produceWithTopology<HcalQIETypes>(record);
}

std::unique_ptr<HcalChannelQuality>
HcalDbProducer::produceChannelQualityWithTopo( const HcalChannelQualityRcd& record) {
  return produceWithTopology<HcalChannelQuality>(record);
}

std::unique_ptr<HcalZSThresholds>
HcalDbProducer::produceZSThresholdsWithTopo(const HcalZSThresholdsRcd& record) {
  return produceWithTopology<HcalZSThresholds>(record);
}

std::unique_ptr<HcalRespCorrs>
HcalDbProducer::produceRespCorrsWithTopo(const HcalRespCorrsRcd& record) {
  return produceWithTopology<HcalRespCorrs>(record);
}

std::unique_ptr<HcalL1TriggerObjects>
HcalDbProducer::produceL1triggerObjectsWithTopo(const HcalL1TriggerObjectsRcd& record) {
  return produceWithTopology<HcalL1TriggerObjects>(record);
}

std::unique_ptr<HcalTimeCorrs>
HcalDbProducer::produceTimeCorrsWithTopo(const HcalTimeCorrsRcd& record) {
  return produceWithTopology<HcalTimeCorrs>(record);
}

std::unique_ptr<HcalLUTCorrs>
HcalDbProducer::produceLUTCorrsWithTopo(const HcalLUTCorrsRcd& record) {
  return produceWithTopology<HcalLUTCorrs>(record);
}

std::unique_ptr<HcalPFCorrs>
HcalDbProducer::producePFCorrsWithTopo(const HcalPFCorrsRcd& record) {
  return produceWithTopology<HcalPFCorrs>(record);
}

std::unique_ptr<HcalLutMetadata>
HcalDbProducer::produceLUTMetadataWithTopo(const HcalLutMetadataRcd& record) {
  return produceWithTopology<HcalLutMetadata>(record);
}

std::unique_ptr<HcalSiPMParameters>
HcalDbProducer::produceSiPMParametersWithTopo(const HcalSiPMParametersRcd& record) {
  return produceWithTopology<HcalSiPMParameters>(record);
}

std::unique_ptr<HcalTPChannelParameters>
HcalDbProducer::produceTPChannelParametersWithTopo(const HcalTPChannelParametersRcd& record) {
  return produceWithTopology<HcalTPChannelParameters>(record);
}

std::unique_ptr<HcalMCParams>
HcalDbProducer::produceMCParamsWithTopo(const HcalMCParamsRcd& record) {
  return produceWithTopology<HcalMCParams>(record);
}

std::unique_ptr<HcalRecoParams>
HcalDbProducer::produceRecoParamsWithTopo(const HcalRecoParamsRcd& record) {
  return produceWithTopology<HcalRecoParams>(record);
}

void HcalDbProducer::setupPedestals(const HcalPedestalsRcd& record,
                                    HcalDbService& service) {
  edm::ESHandle<HcalPedestals> item;
  record.get("withTopo", item);
  service.setData(item.product());

  if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("Pedestals")) != mDumpRequest.end()) {
    *mDumpStream << "New HCAL Pedestals set" << std::endl;
    HcalDbASCIIIO::dumpObject (*mDumpStream, *item);
  }
}

void HcalDbProducer::setupEffectivePedestals(const HcalPedestalsRcd& record,
                                             HcalDbService& service) {
  edm::ESHandle<HcalPedestals> item;
  record.get("withTopoEff", item);
  service.setData(item.product(), true);

  if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("EffectivePedestals")) != mDumpRequest.end()) {
    *mDumpStream << "New HCAL EffectivePedestals set" << std::endl;
    HcalDbASCIIIO::dumpObject (*mDumpStream, *item);
  }
}

void HcalDbProducer::setupPedestalWidths(const HcalPedestalWidthsRcd& record,
                                         HcalDbService& service) {
  edm::ESHandle<HcalPedestalWidths> item;
  record.get("withTopo", item);
  service.setData(item.product());

  if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("PedestalWidths")) != mDumpRequest.end()) {
    *mDumpStream << "New HCAL PedestalWidths set" << std::endl;
    HcalDbASCIIIO::dumpObject (*mDumpStream, *item);
  }
}

void HcalDbProducer::setupEffectivePedestalWidths(const HcalPedestalWidthsRcd& record,
                                                  HcalDbService& service) {
  edm::ESHandle<HcalPedestalWidths> item;
  record.get("withTopoEff", item);
  service.setData(item.product(), true);

  if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("EffectivePedestalWidths")) != mDumpRequest.end()) {
    *mDumpStream << "New HCAL EffectivePedestalWidths set" << std::endl;
    HcalDbASCIIIO::dumpObject (*mDumpStream, *item);
  }
}
