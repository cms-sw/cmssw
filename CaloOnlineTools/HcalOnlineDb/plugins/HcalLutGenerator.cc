// -*- C++ -*-
//
// Package:    CaloOnlineTools/HcalOnlineDb
// Class:      HcalLutGenerator
//
/**\class HcalLutGenerator HcalLutGenerator.cc CaloOnlineTools/HcalOnlineDb/plugins/HcalLutGenerator.cc

 Description: Initializes and commands the HcalLutManager

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Joshua C. Hiltbrand
//         Created:  Thu, 27 Aug 2026 06:15:40 GMT
//

#include "CalibCalorimetry/CaloTPG/interface/CaloTPGTranscoderULUT.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGRecord.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalTPGCoder.h"
#include "CalibFormats/HcalObjects/interface/HcalTPGRecord.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/HcalLutManager.h"
#include "CondFormats/DataRecord/interface/HcalChannelQualityRcd.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <cstdint>
#include <string>

class HcalLutGenerator : public edm::one::EDAnalyzer<> {
public:
  explicit HcalLutGenerator(const edm::ParameterSet&);
  ~HcalLutGenerator() override {}
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override {}

private:
  std::string _tag;
  std::string _lin_file;
  uint32_t _status_word_to_mask;
  edm::ESGetToken<HcalTPGCoder, HcalTPGRecord> tok_inCoder_;
  edm::ESGetToken<HcalDbService, HcalDbRecord> tok_dbservice_;
  edm::ESGetToken<HcalChannelQuality, HcalChannelQualityRcd> tok_hcalChStatus_;
  edm::ESGetToken<CaloTPGTranscoder, CaloTPGRecord> tok_hcalCoder_;
};

HcalLutGenerator::HcalLutGenerator(const edm::ParameterSet& iConfig) {
  _tag = iConfig.getParameter<std::string>("tag");
  _lin_file = iConfig.getParameter<std::string>("HO_master_file");
  _status_word_to_mask = iConfig.getParameter<uint32_t>("status_word_to_mask");
  tok_inCoder_ = esConsumes<HcalTPGCoder, HcalTPGRecord>();
  tok_dbservice_ = esConsumes<HcalDbService, HcalDbRecord>();
  tok_hcalChStatus_ = esConsumes<HcalChannelQuality, HcalChannelQualityRcd>(edm::ESInputTag("", "withTopo"));
  tok_hcalCoder_ = esConsumes<CaloTPGTranscoder, CaloTPGRecord>();
}

void HcalLutGenerator::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const HcalTPGCoder* inputCoder = &iSetup.getData(tok_inCoder_);
  const HcalDbService* hcalcond = &iSetup.getData(tok_dbservice_);
  const HcalChannelQuality* _cq = &iSetup.getData(tok_hcalChStatus_);

  edm::ESHandle<CaloTPGTranscoder> outTranscoder = iSetup.getHandle(tok_hcalCoder_);
  edm::ESHandle<CaloTPGTranscoderULUT> transcoder;
  transcoder.swap(outTranscoder);

  HcalLutManager manager(hcalcond, _cq, _status_word_to_mask);

  manager.createLutXmlFiles_HBEFFromCoder_HOFromAscii_ZDC(_tag, *inputCoder, *transcoder, _lin_file);
}

void HcalLutGenerator::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("tag", "NewLutTag")->setComment("Tag for naming LUT XML file");
  desc.add<std::string>("HO_master_file", "HO_ped9_inputLUTcoderDec.txt")
      ->setComment("HO ascii LUT file to use for building XML");
  desc.add<uint32_t>("status_word_to_mask", 0x8000)
      ->setComment("Channel quality mask for identifying channels to zero LUTs");
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(HcalLutGenerator);
