/** \file
 *
 *  $Date: 2013/05/28 06:00:27 $
 *  $Revision: 1.12 $
 *  \author M. Maggi -- INFN Bari
*/

#include "RPCRecHitProducer.h"


#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"

#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHit.h"

#include "RecoLocalMuon/RPCRecHit/interface/RPCRecHitBaseAlgo.h"
#include "RecoLocalMuon/RPCRecHit/interface/RPCRecHitAlgoFactory.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"

#include "CondFormats/RPCObjects/interface/RPCMaskedStrips.h"
#include "CondFormats/DataRecord/interface/RPCMaskedStripsRcd.h"
#include "CondFormats/RPCObjects/interface/RPCDeadStrips.h"
#include "CondFormats/DataRecord/interface/RPCDeadStripsRcd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>


using namespace edm;
using namespace std;


RPCRecHitProducer::RPCRecHitProducer(const ParameterSet& config){

  // Set verbose output

  produces<RPCRecHitCollection>();

  theRPCDigiLabel = config.getParameter<InputTag>("rpcDigiLabel");
  
  // Get the concrete reconstruction algo from the factory

  string theAlgoName = config.getParameter<string>("recAlgo");
  theAlgo = RPCRecHitAlgoFactory::get()->create(theAlgoName,
						config.getParameter<ParameterSet>("recAlgoConfig"));

  // Get masked- and dead-strip information

  RPCMaskedStripsObj = new RPCMaskedStrips();

  RPCDeadStripsObj = new RPCDeadStrips();

  maskSource = config.getParameter<std::string>("maskSource");

  if (maskSource == "File") {
    edm::FileInPath fp = config.getParameter<edm::FileInPath>("maskvecfile");
    std::ifstream inputFile(fp.fullPath().c_str(), std::ios::in);
    if ( !inputFile ) {
      std::cerr << "Masked Strips File cannot not be opened" << std::endl;
      exit(1);
    }
    while ( inputFile.good() ) {
      RPCMaskedStrips::MaskItem Item;
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
      RPCDeadStrips::DeadItem Item;
      inputFile >> Item.rawId >> Item.strip;
      if ( inputFile.good() ) DeadVec.push_back(Item);
    }
    inputFile.close();
  }

}


RPCRecHitProducer::~RPCRecHitProducer(){

  delete theAlgo;
  delete RPCMaskedStripsObj;
  delete RPCDeadStripsObj;

}



void RPCRecHitProducer::beginRun(const edm::Run& r, const edm::EventSetup& setup){

  // Getting the masked-strip information

  if ( maskSource == "EventSetup" ) {
    edm::ESHandle<RPCMaskedStrips> readoutMaskedStrips;
    setup.get<RPCMaskedStripsRcd>().get(readoutMaskedStrips);
    const RPCMaskedStrips* tmp_obj = readoutMaskedStrips.product();
    RPCMaskedStripsObj->MaskVec = tmp_obj->MaskVec;
    delete tmp_obj;
  }
  else if ( maskSource == "File" ) {
    std::vector<RPCMaskedStrips::MaskItem>::iterator posVec;
    for ( posVec = MaskVec.begin(); posVec != MaskVec.end(); ++posVec ) {
      RPCMaskedStrips::MaskItem Item; 
      Item.rawId = (*posVec).rawId;
      Item.strip = (*posVec).strip;
      RPCMaskedStripsObj->MaskVec.push_back(Item);
    }
  }

  // Getting the dead-strip information

  if ( deadSource == "EventSetup" ) {
    edm::ESHandle<RPCDeadStrips> readoutDeadStrips;
    setup.get<RPCDeadStripsRcd>().get(readoutDeadStrips);
    const RPCDeadStrips* tmp_obj = readoutDeadStrips.product();
    RPCDeadStripsObj->DeadVec = tmp_obj->DeadVec;
    delete tmp_obj;
  }
  else if ( deadSource == "File" ) {
    std::vector<RPCDeadStrips::DeadItem>::iterator posVec;
    for ( posVec = DeadVec.begin(); posVec != DeadVec.end(); ++posVec ) {
      RPCDeadStrips::DeadItem Item;
      Item.rawId = (*posVec).rawId;
      Item.strip = (*posVec).strip;
      RPCDeadStripsObj->DeadVec.push_back(Item);
    }
  }

}



void RPCRecHitProducer::produce(Event& event, const EventSetup& setup) {

  // Get the RPC Geometry

  ESHandle<RPCGeometry> rpcGeom;
  setup.get<MuonGeometryRecord>().get(rpcGeom);

  // Get the digis from the event

  Handle<RPCDigiCollection> digis; 
  event.getByLabel(theRPCDigiLabel,digis);

  // Pass the EventSetup to the algo

  theAlgo->setES(setup);

  // Create the pointer to the collection which will store the rechits

  auto_ptr<RPCRecHitCollection> recHitCollection(new RPCRecHitCollection());

  // Iterate through all digi collections ordered by LayerId   

  RPCDigiCollection::DigiRangeIterator rpcdgIt;
  for (rpcdgIt = digis->begin(); rpcdgIt != digis->end();
       ++rpcdgIt){
       
    // The layerId
    const RPCDetId& rpcId = (*rpcdgIt).first;

    // Get the GeomDet from the setup
    const RPCRoll* roll = rpcGeom->roll(rpcId);
    if (roll == 0){
      edm::LogError("BadDigiInput")<<"Failed to find RPCRoll for ID "<<rpcId;
      continue;
    }

    // Get the iterators over the digis associated with this LayerId
    const RPCDigiCollection::Range& range = (*rpcdgIt).second;


    // Getting the roll mask, that includes dead strips, for the given RPCDet

    RollMask mask;
    int rawId = rpcId.rawId();
    int Size = RPCMaskedStripsObj->MaskVec.size();
    for (int i = 0; i < Size; i++ ) {
      if ( RPCMaskedStripsObj->MaskVec[i].rawId == rawId ) {
	int bit = RPCMaskedStripsObj->MaskVec[i].strip;
	mask.set(bit-1);
      }
    }

    Size = RPCDeadStripsObj->DeadVec.size();
    for (int i = 0; i < Size; i++ ) {
      if ( RPCDeadStripsObj->DeadVec[i].rawId == rawId ) {
	int bit = RPCDeadStripsObj->DeadVec[i].strip;
	mask.set(bit-1);
      }
    }

    // Call the reconstruction algorithm    

    OwnVector<RPCRecHit> recHits =
      theAlgo->reconstruct(*roll, rpcId, range, mask);
    
    if(recHits.size() > 0) //FIXME: is it really needed?
      recHitCollection->put(rpcId, recHits.begin(), recHits.end());
  }

  event.put(recHitCollection);

}

