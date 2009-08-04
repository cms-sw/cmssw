#include "CaloOnlineTools/HcalOnlineDb/interface/HcalLutGenerator.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "CalibFormats/HcalObjects/interface/HcalTPGRecord.h"
#include "CalibFormats/HcalObjects/interface/HcalTPGCoder.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/HcalLutManager.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGRecord.h"
//#include "CalibCalorimetry/CaloTPG/src/CaloTPGTranscoderULUT.h"
 #include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"

#include "CaloOnlineTools/HcalOnlineDb/interface/LMap.h"


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
  
  //
  //_____ get the coders from Event Setup _______________________________
  //
  edm::ESHandle<HcalTPGCoder> inputCoder;
  iSetup.get<HcalTPGRecord>().get(inputCoder);
  HcalTopology theTopo;
  HcalDetId did;
  //
  edm::ESHandle<CaloTPGTranscoder> outTranscoder;
  iSetup.get<CaloTPGRecord>().get(outTranscoder);
  outTranscoder->setup(iSetup,CaloTPGTranscoder::HcalTPG);
  edm::ESHandle<CaloTPGTranscoderULUT> transcoder;
  transcoder.swap(outTranscoder);

  //
  //_____ get EMAP from Event Setup _____________________________________
  //
  edm::ESHandle<HcalElectronicsMap> hEmap;
  iSetup.get<HcalElectronicsMapRcd>().get(hEmap);
  std::vector<HcalGenericDetId> vEmap = hEmap->allPrecisionId();
  cout << "EMAP from Event Setup has " << vEmap.size() << " entries" << endl;

  //EMap _emap(&(*hEmap));

  //
  //_____ generate LUTs _________________________________________________
  //
  //HcalLutManager * manager = new HcalLutManager(); // old ways
  HcalLutManager * manager = new HcalLutManager(&(*hEmap));
  bool split_by_crate = true;
  //manager . createAllLutXmlFilesFromCoder( *inputCoder, _tag, split_by_crate );
  cout << " tag name: " << _tag << endl;
  cout << " HO master file: " << _lin_file << endl;
  //manager -> createLutXmlFiles_HBEFFromCoder_HOFromAscii( _tag, *inputCoder, _lin_file, split_by_crate );
  manager -> createLutXmlFiles_HBEFFromCoder_HOFromAscii( _tag, *inputCoder, *transcoder, _lin_file, split_by_crate );
  delete manager;

  transcoder->releaseSetup();
   
}


void HcalLutGenerator::endJob() {

}
