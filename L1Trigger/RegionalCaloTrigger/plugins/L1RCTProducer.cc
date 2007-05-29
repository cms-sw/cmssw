#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTProducer.h" 

#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGRecord.h"
#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"
#include "CondFormats/DataRecord/interface/L1EmEtScaleRcd.h"

#include "L1Trigger/RegionalCaloTrigger/interface/L1RCT.h"

#include <vector>
using std::vector;

#include <iostream>
using std::cout;
using std::endl;

L1RCTProducer::L1RCTProducer(const edm::ParameterSet& conf) : 
  rct(0),
  src(conf.getParameter<edm::FileInPath>("src")),
  orcaFileInput(conf.getUntrackedParameter<bool>("orcaFileInput")),
  lutFile(conf.getParameter<edm::FileInPath>("lutFile")),
  rctTestInputFile(conf.getParameter<std::string>("rctTestInputFile")),
  rctTestOutputFile(conf.getParameter<std::string>("rctTestOutputFile")),
  patternTest(conf.getUntrackedParameter<bool>("patternTest")),
  lutFile2(conf.getParameter<edm::FileInPath>("lutFile2")),
  useEcal(conf.getParameter<bool>("useEcal")),
  useHcal(conf.getParameter<bool>("useHcal")),
  ecalDigisLabel(conf.getParameter<edm::InputTag>("ecalDigisLabel")),
  hcalDigisLabel(conf.getParameter<edm::InputTag>("hcalDigisLabel"))
{
  //produces<JSCOutput>();
  
  //my try
  // need to include classes for EmCand, Region, and both collections!  done
  //produces<L1CaloEmCollection>("isoEmCollection");
  //produces<L1CaloEmCollection>("nonIsoEmCollection");
  produces<L1CaloEmCollection>();
  produces<L1CaloRegionCollection>();
}

L1RCTProducer::~L1RCTProducer()
{
  if(rct != 0) delete rct;
}

void L1RCTProducer::beginJob(const edm::EventSetup& eventSetup)
{
  if (patternTest)
    {
      rct = new L1RCT(lutFile.fullPath(), lutFile2.fullPath(), rctTestInputFile, rctTestOutputFile,true);
    }
  else
    {
      edm::ESHandle<CaloTPGTranscoder> transcoder;
      eventSetup.get<CaloTPGRecord>().get(transcoder);
      rct = new L1RCT(lutFile.fullPath(), transcoder, rctTestInputFile, rctTestOutputFile);
    }
}

void L1RCTProducer::produce(edm::Event& e, const edm::EventSetup& c)
{
  //vector<vector<vector<unsigned short> > > barrel;
  //vector<vector<unsigned short> > hf;
  
  if (!orcaFileInput){
    
    edm::Handle<EcalTrigPrimDigiCollection> ecal;
    edm::Handle<HcalTrigPrimDigiCollection> hcal;
    edm::ESHandle<L1CaloEtScale> emScale;
    
    //if (useEcal) { e.getByLabel("ecalTriggerPrimitiveDigis",ecal); }
    //if (useHcal) { e.getByLabel("hcalTriggerPrimitiveDigis",hcal); }
    if (useEcal) { e.getByLabel(ecalDigisLabel, ecal); }
    if (useHcal) { e.getByLabel(hcalDigisLabel, hcal); }
    c.get<L1EmEtScaleRcd>().get(emScale);

    if (emScale.product() != 0) {
      rct->setGctEmScale(emScale.product());
    }
    
    EcalTrigPrimDigiCollection ecalColl;
    HcalTrigPrimDigiCollection hcalColl;
    if (ecal.isValid()) { ecalColl = *ecal; }
    if (hcal.isValid()) { hcalColl = *hcal; }

    //rct->digiInput(*ecal, *hcal);
    rct->digiInput(ecalColl, hcalColl);

  }
  
  else if (orcaFileInput){
    // Only for inputting directly from file
    //rct->fileInput("../data/rct-input-1.dat");
    //const char* filename = src.c_str();
    //std::cout << "filename is " << filename << endl;
    rct->fileInput(src.fullPath().c_str());
    //rct->fileInput(filename);
    //cout << "L1RCT: file has been input" << endl;
  } 

  rct->processEvent();
  //rct->printJSC();
  
  
  // Stuff to create
  /*
  std::auto_ptr<L1CaloEmCollection> rctIsoEmCands (new L1CaloEmCollection);
  std::auto_ptr<L1CaloEmCollection> rctNonIsoEmCands (new L1CaloEmCollection);
  */
  // retry
  std::auto_ptr<L1CaloEmCollection> rctEmCands (new L1CaloEmCollection);

  //fill these above?  like gct:
  for (int j = 0; j<18; j++){
    for (int i = 0; i<4; i++) {
      /*
      if (j == 0){
	cout << "\n\nPrinting EGObjects " << i << " for crate " << j <<endl;
	cout << rct->getIsolatedEGObjects(j).at(i) << endl;
	cout << rct->getNonisolatedEGObjects(j).at(i) << endl;
      }
      */
      rctEmCands->push_back(rct->getIsolatedEGObjects(j).at(i));  // or something
      rctEmCands->push_back(rct->getNonisolatedEGObjects(j).at(i));
    }
  }
  

  
  std::auto_ptr<L1CaloRegionCollection> rctRegions (new L1CaloRegionCollection);
  // need to fill this, too
  // if possible, can do by constructor, like in gct:
  //std::auto_ptr<L1GctEtHad> etHadResult(new L1GctEtHad(m_gct->getEtHad().value(), false) );
  // or just do like em objects, using 0-17 for loop and then 0-21 for loop: each region in each crate, like
  for (int i = 0; i < 18; i++){
    vector<L1CaloRegion> regions = rct->getRegions(i);
    for (int j = 0; j < 22; j++){
      /*
      if (i == 0){
        cout << "\n\nPrinting region " << j << " for crate " << i << endl;
	cout << regions.at(j) << endl;
      }
      */
      rctRegions->push_back(regions.at(j));
    }
  }
  


  
  //putting stuff back into event
  e.put(rctEmCands);
  e.put(rctRegions);

}

//#include "FWCare/PluginManager/interface/ModuleDef.h"
//#include "FWCore/Framework/interface/MakerMacros.h"
//DEFINE_SEAL_MODULE();
//DEFINE_ANOTHER_FWK_MODULE(L1RCTProducer);
