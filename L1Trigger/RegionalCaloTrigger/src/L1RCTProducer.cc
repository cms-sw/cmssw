#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTProducer.h"

L1RCTProducer::L1RCTProducer(const edm::ParameterSet& conf) : 
  src(conf.getParameter<string>("src")), 
  orcaFileInput(conf.getUntrackedParameter<bool>("orcaFileInput")),
  lutFile(conf.getParameter<string>("lutFile"))
{
  //produces<JSCOutput>();

  //my try
  // need to include classes for EmCand, Region, and both collections!  done
  //produces<L1CaloEmCollection>("isoEmCollection");
  //produces<L1CaloEmCollection>("nonIsoEmCollection");
  produces<L1CaloEmCollection>();
  produces<L1CaloRegionCollection>();

  rct = new L1RCT(lutFile);
  //std::cout << "One L1RCTProducer constructed!" << std::endl;
}

L1RCTProducer::~L1RCTProducer(){
  delete rct;
  //std::cout << "One L1RCTProducer deleted!" << std::endl;
}

void L1RCTProducer::produce(edm::Event& e, const edm::EventSetup& c)
{
  //vector<vector<vector<unsigned short> > > barrel;
  //vector<vector<unsigned short> > hf;
  
  //std::cout << "produce method entered" << std::endl;

  if (!orcaFileInput){
    // my try:
    edm::Handle<EcalTrigPrimDigiCollection> ecal;
    edm::Handle<HcalTrigPrimDigiCollection> hcal;
    edm::ESHandle<L1CaloEtScale> emScale;
    e.getByType(ecal);
    //e.getByType(hcal);
    e.getByLabel("hcalTriggerPrimitiveDigis",hcal);
    c.get<L1EmEtScaleRcd>().get(emScale);

    // as in L1GctEmulator.cc
    if (emScale.product() != 0) {
      rct->setGctEmScale(emScale.product());
    }
    rct->digiInput(*ecal, *hcal);
  }
  
  else if (orcaFileInput){
    // Only for inputting directly from file
    //rct->fileInput("../data/rct-input-1.dat");
    //const char* filename = src.c_str();
    //std::cout << "filename is " << filename << endl;
    rct->fileInput(src.c_str());
    //rct->fileInput(filename);
    //std::cout << "file has been inputted" << std::endl;
  } 

  rct->processEvent();
  //std::cout << "event has been processed" << std::endl;
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

//#include "PluginManager/ModuleDef.h"
//#include "FWCore/Framework/interface/MakerMacros.h"
//DEFINE_SEAL_MODULE();
//DEFINE_ANOTHER_FWK_MODULE(L1RCTProducer);
