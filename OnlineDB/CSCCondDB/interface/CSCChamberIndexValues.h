#ifndef _CSCCHAMBERINDEXVALUES_H
#define _CSCCHAMBERINDEXVALUES_H

#include <memory>
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include "CondFormats/CSCObjects/interface/CSCChamberIndex.h"
#include "CondFormats/DataRecord/interface/CSCChamberIndexRcd.h"
#include "OnlineDB/CSCCondDB/interface/CSCChamberIndexValues.h"
#include "CondFormats/CSCObjects/interface/CSCMapItem.h"
#include "OnlineDB/CSCCondDB/interface/CSCMap1.h"

class CSCChamberIndexValues: public edm::ESProducer, public edm::EventSetupRecordIntervalFinder  {
 public:
  CSCChamberIndexValues(const edm::ParameterSet&);
  ~CSCChamberIndexValues();
  
  inline static CSCChamberIndex * fillChamberIndex();

  typedef const  CSCChamberIndex * ReturnType;
  
  ReturnType produceChamberIndex(const CSCChamberIndexRcd&);
  
 private:
  // ----------member data ---------------------------
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&, edm::ValidityInterval & );
  CSCChamberIndex *mapObj ;

};

#include<fstream>
#include<vector>
#include<iostream>

// to workaround plugin library
inline CSCChamberIndex *  CSCChamberIndexValues::fillChamberIndex()
{
  CSCChamberIndex * mapobj = new CSCChamberIndex();
  cscmap1 *map = new cscmap1();
  CSCMapItem::MapItem item;
  int chamberid;

  int i,j,k,l; //i - endcap, j - station, k - ring, l - chamber.
  int r,c;     //r - number of rings, c - number of chambers.
  int count=0;

  mapobj->ch_index.resize(540);
  /* This is version for 540 chambers. */
  for(i=1;i<=2;++i){
    for(j=1;j<=4;++j){
      if(j==1) r=3;
      //else if(j==4) r=1;
      else r=2;
      for(k=1;k<=r;++k){
       if(j>1 && k==1) c=18;
       else c=36;
        for(l=1;l<=c;++l){
         chamberid=i*100000+j*10000+k*1000+l*10;
         map->chamber(chamberid,&item);
         mapobj->ch_index[item.cscIndex-1]=item;
         count=count+1;
        }
      }
    }
  }
  return mapobj;
}

#endif
