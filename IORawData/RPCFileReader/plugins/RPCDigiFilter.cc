
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"

#include "IORawData/RPCFileReader/interface/RPCDigiFilter.h"
#include <DataFormats/RPCDigi/interface/RPCDigiCollection.h>

#include "CondFormats/RPCObjects/interface/RPCReadOutMapping.h"
#include "CondFormats/DataRecord/interface/RPCReadOutMappingRcd.h"
#include "CondFormats/RPCObjects/interface/ChamberLocationSpec.h"
#include "CondFormats/RPCObjects/interface/LinkBoardElectronicIndex.h"


RPCDigiFilter::RPCDigiFilter(const edm::ParameterSet& ps) {

  produces<RPCDigiCollection>();
}


RPCDigiFilter::~RPCDigiFilter() {}


void RPCDigiFilter::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

  // Create empty output
  std::auto_ptr<RPCDigiCollection> pDigis(new RPCDigiCollection());

  ///Read RPC digis
  edm::Handle<RPCDigiCollection> rpcDigis;
  iEvent.getByType(rpcDigis);
  RPCDigiCollection::DigiRangeIterator rpcDigiCI;
  for(rpcDigiCI = rpcDigis->begin();rpcDigiCI!=rpcDigis->end();rpcDigiCI++){
    //check if the chamber is present in the cabling DB.
    int rawDetId = (*rpcDigiCI).first.rawId();
    int stripInDU =  (*rpcDigiCI).second.first->strip();
    bool accept = acceptDigiDB(std::pair<uint32_t, int>(rawDetId, stripInDU), iSetup);
    bool accept1 = acceptDigiGeom(std::pair<uint32_t, int>(rawDetId, stripInDU));
    if(accept!=accept1) {
      //std::cout<<"Difference: accept from DB: "<<accept<<" accept from geom: "<<accept1<<" digi: "<<(*rpcDigiCI).first;
      //std::cout<<"     detId: "<< rawDetId<<" strip: "<< stripInDU<<std::endl;
    }
    if(!accept){
      //std::cout<<"Rejected digi for chamber: "<<(*rpcDigiCI).first<<std::endl;
      continue;
    }
    const RPCDigiCollection::Range& range = (*rpcDigiCI).second;    
    for (RPCDigiCollection::const_iterator digiIt = range.first;
         digiIt!=range.second;++digiIt){
      //std::cout<<*digiIt<<std::endl;
      pDigis->insertDigi((*rpcDigiCI).first,*digiIt);     
    }
  }

 
  //grep -A 10 "Re 1 Ri 3 St 1 Se 6 La 1 Su 4 Ro 2 Tr 0" mapMB.out

  // store them in the event
  iEvent.put(pDigis);
}


bool RPCDigiFilter::acceptDigiDB(std::pair<uint32_t, int> detStripPair,  const edm::EventSetup& iSetup){

  //Open the cabling database
  edm::ESHandle<RPCReadOutMapping> readoutMapping;
  iSetup.get<RPCReadOutMappingRcd>().get(readoutMapping);
  
  // decode digi<->map
  typedef std::vector< std::pair< LinkBoardElectronicIndex, LinkBoardPackedStrip> > RawDataFrames;
  RawDataFrames rawDataFrames = readoutMapping->rawDataFrame(detStripPair);
  //std::cout<<"rawDataFrames.size(): "<<rawDataFrames.size()<<std::endl;
  //if(!rawDataFrames.size()) return false;
  
  return (rawDataFrames.size()>0);

}


bool RPCDigiFilter::acceptDigiGeom(std::pair<uint32_t, int> detStripPair){

  RPCDetId aId(detStripPair.first);

  if(aId.region()!=0){
    if(aId.ring()==1) return false; //Reject digis in RE X/1
    if(aId.station()==4) return false; //Reject digis in RE 4/X
  }

    return true;
}
