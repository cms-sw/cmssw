#include "CaloOnlineTools/HcalOnlineDb/interface/HcalLutGenerator.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "CalibFormats/HcalObjects/interface/HcalTPGRecord.h"
#include "CalibFormats/HcalObjects/interface/HcalTPGCoder.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/HcalLutManager.h"


#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

HcalLutGenerator::HcalLutGenerator(const edm::ParameterSet& iConfig)
{
  cout << " --> HcalLutGenerator::HcalLutGenerator()" << endl;
  _tag = iConfig.getParameter<string>("tag");
  _lin_file = iConfig.getParameter<string>("HO_master_file");
}

HcalLutGenerator::~HcalLutGenerator()
{
}


void HcalLutGenerator::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  
  cout << " --> HcalLutGenerator::analyze()" << endl;

  edm::ESHandle<HcalTPGCoder> inputCoder;
  iSetup.get<HcalTPGRecord>().get(inputCoder);
  HcalTopology theTopo;
  HcalDetId did;


  HcalLutManager manager;
  bool split_by_crate = true;
  //manager . createAllLutXmlFilesFromCoder( *inputCoder, _tag, split_by_crate );
  cout << " tag name: " << _tag << endl;
  cout << " HO master file: " << _lin_file << endl;
  manager . createLutXmlFiles_HBEFFromCoder_HOFromAscii( _tag, *inputCoder, _lin_file, split_by_crate );


  // FIXME: compr LUTs off EventSetup - implement here
  //edm::ESHandle<CaloTPGTranscoder> transcoder;
  //iSetup.get<CaloTPGRecord>().get(transcoder);  
  //edm::Handle<HcalTrigPrimDigiCollection> hcal;
  //iEvent.getByLabel("hcalTriggerPrimitiveDigis",hcal);
  //HcalTrigPrimDigiCollection hcalCollection = *hcal;
  //HcalTrigTowerGeometry theTrigTowerGeometry;
   
  // DEBUG: checking a lin LUT
  /*
  did=HcalDetId(HcalBarrel,1,1,1);
  if (theTopo.valid(did)) {
    std::vector<unsigned short> lut=inputCoder->getLinearizationLUT(HcalDetId(did));
    for (std::vector<unsigned short>::const_iterator _i=lut.begin(); _i!=lut.end();_i++){
      unsigned int _entry = (unsigned int)(*_i);
      std::cout << "LUT" << "     " << _entry << std::endl;
      
    }
  }
  */
}


void HcalLutGenerator::endJob() {

}
