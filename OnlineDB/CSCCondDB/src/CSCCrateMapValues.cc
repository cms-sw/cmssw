#include <memory>
#include "boost/shared_ptr.hpp"
#include <fstream>

#include "CondFormats/CSCObjects/interface/CSCCrateMap.h"
#include "CondFormats/DataRecord/interface/CSCCrateMapRcd.h"
#include "OnlineDB/CSCCondDB/interface/CSCCrateMapValues.h"
#include "OnlineDB/CSCCondDB/interface/CSCMap1.h"

void CSCCrateMapValues::fillCrateMap()
{
  mapobj = new CSCCrateMap();
  cscmap1 *map = new cscmap1 ();
  CSCMapItem::MapItem item;

  int i,j,k,l; //i - endcap, j - station, k - ring, l - chamber.
  int r,c;     //r - number of rings, c - number of chambers.
  int count=0;
  int chamberid;
  int crate_cscid;

  for(i=1;i<=2;++i){
    for(j=1;j<=4;++j){
      if(j==1) r=3;
      else if(j==4) r=1;
      else r=2;
      for(k=1;k<=r;++k){
       if(j>1 && k==1) c=18;
       else c=36;
        for(l=1;l<=c;++l){
         chamberid=i*100000+j*10000+k*1000+l*10;
         map->chamber(chamberid,&item);
         crate_cscid=item.crateid*10+item.cscid;
         mapobj->crate_map[crate_cscid]=item;
         count=count+1;
        }
      }
    }
  }
}
  

CSCCrateMapValues::CSCCrateMapValues(const edm::ParameterSet& iConfig)
{
  //the following line is needed to tell the framework what
  // data is being produced
  fillCrateMap();
  setWhatProduced(this,&CSCCrateMapValues::produceCrateMap);
  findingRecord<CSCCrateMapRcd>();
  //now do what ever other initialization is needed
}


CSCCrateMapValues::~CSCCrateMapValues()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  delete mapobj;
}


//
// member functions
//

// ------------ method called to produce the data  ------------
CSCCrateMapValues::ReturnType
CSCCrateMapValues::produceCrateMap(const CSCCrateMapRcd& iRecord)
{
  //need a new object so to not be deleted at exit
  CSCCrateMap* mydata=new CSCCrateMap( *mapobj );
  return mydata;
  
}

 void CSCCrateMapValues::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&,
 edm::ValidityInterval & oValidity)
 {
 oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),edm::IOVSyncValue::endOfTime());
 
 }
