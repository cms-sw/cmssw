#ifndef _CSCBADSTRIPSCONDITIONS_H
#define _CSCBADSTRIPSCONDITIONS_H

#include <memory>
#include <cmath>
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
#include "CondFormats/CSCObjects/interface/CSCBadStrips.h"
#include "CondFormats/DataRecord/interface/CSCBadStripsRcd.h"

class CSCBadStripsConditions: public edm::ESProducer, public edm::EventSetupRecordIntervalFinder  {
 public:
  CSCBadStripsConditions(const edm::ParameterSet&);
  ~CSCBadStripsConditions();
  

  inline static CSCBadStrips *  prefillBadStrips();

  typedef const  CSCBadStrips * ReturnType;
  
  ReturnType produceBadStrips(const CSCBadStripsRcd&);
  
 private:
  // ----------member data ---------------------------
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&, edm::ValidityInterval & );
  CSCBadStrips *cndbBadStrips ;

};

#include<fstream>
#include<vector>
#include<iostream>

// to workaround plugin library
inline CSCBadStrips *  CSCBadStripsConditions::prefillBadStrips()
{
  //  const int MAX_SIZE = 217728;
  CSCBadStrips * cndbbadstrips = new CSCBadStrips();
  bool bad=false; //use boolean to fill bad channels

  int new_index;
  int new_layer,new_channel, new_flag1, new_cham,new_pointer;
  std::vector<int> new_index_id;
  std::vector<short int> new_layer_id;
  std::vector<short int> new_chan_id;
  std::vector<short int> new_flag_id;
  std::vector<int> new_cham_id;
  std::vector<int> new_point;

  // int counter;
  //  int db_nrlines=0;
  int new_nrlines=0;
 
  std::ifstream newdata;
  newdata.open("badstrips.dat",std::ios::in); 
  if(!newdata) {
    std::cerr <<"Error: badstrips.dat -> no such file!"<< std::endl;
    exit(1);
  }
  
  while (!newdata.eof() ) { 
    newdata >> new_index >> new_layer >> new_channel >> new_flag1 >> new_cham >> new_pointer; 
    new_index_id.push_back(new_index);
    new_layer_id.push_back(new_layer);
    new_chan_id.push_back(new_channel);
    new_flag_id.push_back(new_flag1);
    new_cham_id.push_back(new_cham);
    new_point.push_back(new_pointer);
    new_nrlines++;
  }
  newdata.close();
  

  CSCBadStrips::BadChannelContainer & itemvector = cndbbadstrips->channels;
  itemvector.resize(new_nrlines);
  CSCBadStrips::BadChamberContainer & itemvector1 = cndbbadstrips->chambers;
  itemvector1.resize(new_nrlines);
  cndbbadstrips->numberOfBadChannels = new_nrlines;

  if(new_index_id.empty()) bad=false; 
  if(!new_index_id.empty()) bad=true;

  for(unsigned int i=0; i<new_index_id.size()-1;++i){
       itemvector[i].layer =  new_layer;
       itemvector[i].channel =  new_channel;
       itemvector[i].flag1 =  new_flag1;
       itemvector1[i].chamber_index = new_cham;
       itemvector1[i].pointer = new_pointer;
       itemvector1[i].bad_channels = new_index;
  }

   return cndbbadstrips;
}

#endif

