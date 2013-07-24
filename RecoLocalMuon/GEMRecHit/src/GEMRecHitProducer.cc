/** \file
 *
 *  $Date: 2013/04/24 17:16:35 $
 *  $Revision: 1.1 $
 *  \author M. Maggi -- INFN Bari
*/

#include "GEMRecHitProducer.h"


#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"

#include "Geometry/GEMGeometry/interface/GEMEtaPartition.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/GEMRecHit/interface/GEMRecHit.h"

#include "RecoLocalMuon/GEMRecHit/interface/GEMRecHitBaseAlgo.h"
#include "RecoLocalMuon/GEMRecHit/interface/GEMRecHitAlgoFactory.h"
#include "DataFormats/GEMRecHit/interface/GEMRecHitCollection.h"

// #include "CondFormats/GEMObjects/interface/GEMMaskedStrips.h"
// #include "CondFormats/DataRecord/interface/GEMMaskedStripsRcd.h"
// #include "CondFormats/GEMObjects/interface/GEMDeadStrips.h"
// #include "CondFormats/DataRecord/interface/GEMDeadStripsRcd.h"

#include <string>


using namespace edm;
using namespace std;


GEMRecHitProducer::GEMRecHitProducer(const ParameterSet& config){

  // Set verbose output

  produces<GEMRecHitCollection>();

  theGEMDigiLabel = config.getParameter<InputTag>("gemDigiLabel");
  
  // Get the concrete reconstruction algo from the factory

  string theAlgoName = config.getParameter<string>("recAlgo");
  theAlgo = GEMRecHitAlgoFactory::get()->create(theAlgoName,
						config.getParameter<ParameterSet>("recAlgoConfig"));

  // Get masked- and dead-strip information

  /* GEMMaskedStripsObj = new GEMMaskedStrips();

  GEMDeadStripsObj = new GEMDeadStrips();

  maskSource = config.getParameter<std::string>("maskSource");

  if (maskSource == "File") {
    edm::FileInPath fp = config.getParameter<edm::FileInPath>("maskvecfile");
    std::ifstream inputFile(fp.fullPath().c_str(), std::ios::in);
    if ( !inputFile ) {
      std::cerr << "Masked Strips File cannot not be opened" << std::endl;
      exit(1);
    }
    while ( inputFile.good() ) {
      GEMMaskedStrips::MaskItem Item;
      inputFile >> Item.rawId >> Item.strip;
      if ( inputFile.good() ) MaskVec.push_back(Item);
    }
    inputFile.close();
  }

  deadSource = config.getParameter<std::string>("deadSource");

  if (deadSource == "File") {
    edm::FileInPath fp = config.getParameter<edm::FileInPath>("deadvecfile");
    std::ifstream inputFile(fp.fullPath().c_str(), std::ios::in);
    if ( !inputFile ) {
      std::cerr << "Dead Strips File cannot not be opened" << std::endl;
      exit(1);
    }
    while ( inputFile.good() ) {
      GEMDeadStrips::DeadItem Item;
      inputFile >> Item.rawId >> Item.strip;
      if ( inputFile.good() ) DeadVec.push_back(Item);
    }
    inputFile.close();
  }
  */
}


GEMRecHitProducer::~GEMRecHitProducer(){

  delete theAlgo;
  // delete GEMMaskedStripsObj;
  // delete GEMDeadStripsObj;

}



void GEMRecHitProducer::beginRun(const edm::Run& r, const edm::EventSetup& setup){

  // Getting the masked-strip information
  /*
  if ( maskSource == "EventSetup" ) {
    edm::ESHandle<GEMMaskedStrips> readoutMaskedStrips;
    setup.get<GEMMaskedStripsRcd>().get(readoutMaskedStrips);
    const GEMMaskedStrips* tmp_obj = readoutMaskedStrips.product();
    GEMMaskedStripsObj->MaskVec = tmp_obj->MaskVec;
    delete tmp_obj;
  }
  else if ( maskSource == "File" ) {
    std::vector<GEMMaskedStrips::MaskItem>::iterator posVec;
    for ( posVec = MaskVec.begin(); posVec != MaskVec.end(); ++posVec ) {
      GEMMaskedStrips::MaskItem Item; 
      Item.rawId = (*posVec).rawId;
      Item.strip = (*posVec).strip;
      GEMMaskedStripsObj->MaskVec.push_back(Item);
    }
  }
  */
  // Getting the dead-strip information
  /*
  if ( deadSource == "EventSetup" ) {
    edm::ESHandle<GEMDeadStrips> readoutDeadStrips;
    setup.get<GEMDeadStripsRcd>().get(readoutDeadStrips);
    const GEMDeadStrips* tmp_obj = readoutDeadStrips.product();
    GEMDeadStripsObj->DeadVec = tmp_obj->DeadVec;
    delete tmp_obj;
  }
  else if ( deadSource == "File" ) {
    std::vector<GEMDeadStrips::DeadItem>::iterator posVec;
    for ( posVec = DeadVec.begin(); posVec != DeadVec.end(); ++posVec ) {
      GEMDeadStrips::DeadItem Item;
      Item.rawId = (*posVec).rawId;
      Item.strip = (*posVec).strip;
      GEMDeadStripsObj->DeadVec.push_back(Item);
    }
  }
  */
}



void GEMRecHitProducer::produce(Event& event, const EventSetup& setup) {

  // Get the GEM Geometry

  ESHandle<GEMGeometry> gemGeom;
  setup.get<MuonGeometryRecord>().get(gemGeom);

  // Get the digis from the event

  Handle<GEMDigiCollection> digis; 
  event.getByLabel(theGEMDigiLabel,digis);

  // Pass the EventSetup to the algo

  theAlgo->setES(setup);

  // Create the pointer to the collection which will store the rechits

  auto_ptr<GEMRecHitCollection> recHitCollection(new GEMRecHitCollection());

  // Iterate through all digi collections ordered by LayerId   

  GEMDigiCollection::DigiRangeIterator gemdgIt;
  for (gemdgIt = digis->begin(); gemdgIt != digis->end();
       ++gemdgIt){
       
    // The layerId
    const GEMDetId& gemId = (*gemdgIt).first;

    // Get the GeomDet from the setup
    const GEMEtaPartition* roll = gemGeom->etaPartition(gemId);

    // Get the iterators over the digis associated with this LayerId
    const GEMDigiCollection::Range& range = (*gemdgIt).second;


    // Getting the roll mask, that includes dead strips, for the given GEMDet
    EtaPartitionMask mask;
    /*
    int rawId = gemId.rawId();
    int Size = GEMMaskedStripsObj->MaskVec.size();
    for (int i = 0; i < Size; i++ ) {
      if ( GEMMaskedStripsObj->MaskVec[i].rawId == rawId ) {
	int bit = GEMMaskedStripsObj->MaskVec[i].strip;
	mask.set(bit-1);
      }
    }

    Size = GEMDeadStripsObj->DeadVec.size();
    for (int i = 0; i < Size; i++ ) {
      if ( GEMDeadStripsObj->DeadVec[i].rawId == rawId ) {
	int bit = GEMDeadStripsObj->DeadVec[i].strip;
	mask.set(bit-1);
      }
    }
    */
    // Call the reconstruction algorithm    

    OwnVector<GEMRecHit> recHits =
      theAlgo->reconstruct(*roll, gemId, range, mask);
    
    if(recHits.size() > 0) //FIXME: is it really needed?
      recHitCollection->put(gemId, recHits.begin(), recHits.end());
  }

  event.put(recHitCollection);

}

