#include <memory>
#include "boost/shared_ptr.hpp"
#include <fstream>

#include "CondFormats/CSCObjects/interface/CSCChamberMap.h"
#include "CondFormats/DataRecord/interface/CSCChamberMapRcd.h"
#include "OnlineDB/CSCCondDB/interface/CSCChamberMapValues.h"
#include "OnlineDB/CSCCondDB/interface/cscmap1.h"

void CSCChamberMapValues::fillChamberMap()
{
  mapobj = new CSCChamberMap();
  cscmap1 *map = new cscmap1 ();
  CSCMapItem::MapItem item;

  int i,j,k,l; //i - endcap, j - station, k - ring, l - chamber.
  int r,c;     //r - number of rings, c - number of chambers.
  int count=0;
  int chamberid;

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
	 //         mapobj->ch_map[chamberid].chamberLabel=item.chamberLabel;
         mapobj->ch_map[chamberid]=item;
         count=count+1;
         //std::cout<<count<<") ";
	 //std::cout<<i<<" "<<j<<" "<<k<<" "<<l<<" "<<r<<" "<<c<<"  ";
	 //std::cout<<chamberid<<"  ";
	 //std::cout<<item.chamberId<<"  ";
	 //std::cout<<item.chamberLabel<<"  ";
	 //std::cout<<item.cscIndex<<"  ";
	 //std::cout<<item.crateid<<"  ";
	 //std::cout<<item.strips<<"  ";
	 //std::cout<<item.sector<<"  ";
	 //std::cout<<item.ddu<<"-"<<item.ddu_input;
         //std::cout<<std::endl;
        }
      }
    }
    //std::cout<<"Print\n";
  }
  //std::cout<<232350<<"  ";
  //std::cout<<mapobj->ch_map[232350].chamberLabel<<"  ";
  //std::cout<<mapobj->ch_map[232350].cscIndex<<"  ";
  //std::cout<<std::endl;
}
  

CSCChamberMapValues::CSCChamberMapValues(const edm::ParameterSet& iConfig)
{
  //the following line is needed to tell the framework what
  // data is being produced
  fillChamberMap();
  setWhatProduced(this,&CSCChamberMapValues::produceChamberMap);
  findingRecord<CSCChamberMapRcd>();
  //now do what ever other initialization is needed
}


CSCChamberMapValues::~CSCChamberMapValues()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  delete mapobj;
}


//
// member functions
//

// ------------ method called to produce the data  ------------
CSCChamberMapValues::ReturnType
CSCChamberMapValues::produceChamberMap(const CSCChamberMapRcd& iRecord)
{
  //need a new object so to not be deleted at exit
  CSCChamberMap* mydata=new CSCChamberMap( *mapobj );
  return mydata;
  
}

 void CSCChamberMapValues::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&,
 edm::ValidityInterval & oValidity)
 {
 oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),edm::IOVSyncValue::endOfTime());
 
 }
