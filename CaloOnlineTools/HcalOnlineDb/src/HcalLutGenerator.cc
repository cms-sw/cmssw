#include "CaloOnlineTools/HcalOnlineDb/interface/HcalLutGenerator.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/HcalLutManager.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGRecord.h"

#include "CaloOnlineTools/HcalOnlineDb/interface/LMap.h"

#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

HcalLutGenerator::HcalLutGenerator(const edm::ParameterSet& iConfig) {
  _tag = iConfig.getParameter<std::string>("tag");
  _lin_file = iConfig.getParameter<std::string>("HO_master_file");
  _status_word_to_mask = iConfig.getParameter<uint32_t>("status_word_to_mask");
  tok_inCoder_ = esConsumes<HcalTPGCoder, HcalTPGRecord>();
  tok_dbservice_ = esConsumes<HcalDbService, HcalDbRecord>();
  tok_hcalChStatus_ = esConsumes<HcalChannelQuality, HcalChannelQualityRcd>(edm::ESInputTag("", "withTopo"));
}

HcalLutGenerator::~HcalLutGenerator() {}

void HcalLutGenerator::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const HcalTPGCoder* inputCoder = &iSetup.getData(tok_inCoder_);
  const HcalDbService* hcalcond = &iSetup.getData(tok_dbservice_);
  const HcalChannelQuality* _cq = &iSetup.getData(tok_hcalChStatus_);

  edm::ESHandle<CaloTPGTranscoder> outTranscoder;
  iSetup.get<CaloTPGRecord>().get(outTranscoder);

  edm::ESHandle<CaloTPGTranscoderULUT> transcoder;
  transcoder.swap(outTranscoder);

  HcalLutManager manager(hcalcond, _cq, _status_word_to_mask);
  bool split_by_crate = true;

  manager.createLutXmlFiles_HBEFFromCoder_HOFromAscii_ZDC(_tag, *inputCoder, *transcoder, _lin_file, split_by_crate);
}

void HcalLutGenerator::endJob() {}
