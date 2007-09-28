#include <memory>
#include "boost/shared_ptr.hpp"
#include <fstream>

#include "CondFormats/CSCObjects/interface/CSCChamberIndex.h"
#include "CondFormats/DataRecord/interface/CSCChamberIndexRcd.h"
#include "OnlineDB/CSCCondDB/interface/CSCChamberIndexValues.h"
#include "OnlineDB/CSCCondDB/interface/CSCMap1.h"

void CSCChamberIndexValues::fillChamberIndex()
{
  mapobj = new CSCChamberIndex();
  cscmap1 *map = new cscmap1 ();
  CSCMapItem::MapItem item;
  int chamberid;

  int i,j,k,l; //i - endcap, j - station, k - ring, l - chamber.
  int r,c;     //r - number of rings, c - number of chambers.
  int count=0;

  mapobj->ch_index.resize(469);
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
         mapobj->ch_index[item.cscIndex]=item;
         count=count+1;
        }
      }
    }
  }
}
  

CSCChamberIndexValues::CSCChamberIndexValues(const edm::ParameterSet& iConfig)
{
  //the following line is needed to tell the framework what
  // data is being produced
  fillChamberIndex();
  setWhatProduced(this,&CSCChamberIndexValues::produceChamberIndex);
  findingRecord<CSCChamberIndexRcd>();
  //now do what ever other initialization is needed
}


CSCChamberIndexValues::~CSCChamberIndexValues()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  delete mapobj;
}


//
// member functions
//

// ------------ method called to produce the data  ------------
CSCChamberIndexValues::ReturnType
CSCChamberIndexValues::produceChamberIndex(const CSCChamberIndexRcd& iRecord)
{
  //need a new object so to not be deleted at exit
  CSCChamberIndex* mydata=new CSCChamberIndex( *mapobj );
  return mydata;
  
}

 void CSCChamberIndexValues::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&,
 edm::ValidityInterval & oValidity)
 {
 oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),edm::IOVSyncValue::endOfTime());
 
 }
