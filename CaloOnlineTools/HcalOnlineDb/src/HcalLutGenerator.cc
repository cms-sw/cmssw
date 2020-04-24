#include "CaloOnlineTools/HcalOnlineDb/interface/HcalLutGenerator.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "CalibFormats/HcalObjects/interface/HcalTPGRecord.h"
#include "CalibFormats/HcalObjects/interface/HcalTPGCoder.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/HcalLutManager.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGRecord.h"
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"

#include "CaloOnlineTools/HcalOnlineDb/interface/LMap.h"


#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

HcalLutGenerator::HcalLutGenerator(const edm::ParameterSet& iConfig)
{
  _tag                 = iConfig.getParameter<std::string>("tag");
  _lin_file            = iConfig.getParameter<std::string>("HO_master_file");
  _status_word_to_mask = iConfig.getParameter<uint32_t>("status_word_to_mask");
}

HcalLutGenerator::~HcalLutGenerator()
{
}

void HcalLutGenerator::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  
  edm::ESHandle<HcalTPGCoder> inputCoder;
  iSetup.get<HcalTPGRecord>().get(inputCoder);

  edm::ESHandle<CaloTPGTranscoder> outTranscoder;
  iSetup.get<CaloTPGRecord>().get(outTranscoder);
 
  edm::ESHandle<CaloTPGTranscoderULUT> transcoder;
  transcoder.swap(outTranscoder);

  edm::ESHandle<HcalDbService> hcalcond;			
  iSetup.get<HcalDbRecord>().get(hcalcond);

  edm::ESHandle<HcalChannelQuality> hCQ;
  iSetup.get<HcalChannelQualityRcd>().get("withTopo",hCQ);
  const HcalChannelQuality * _cq = &(*hCQ);

  HcalLutManager manager(hcalcond.product(), _cq, _status_word_to_mask);
  bool split_by_crate = true;
   
  manager.createLutXmlFiles_HBEFFromCoder_HOFromAscii_ZDC( _tag, *inputCoder, *transcoder, _lin_file, split_by_crate );
}


void HcalLutGenerator::endJob() {}
