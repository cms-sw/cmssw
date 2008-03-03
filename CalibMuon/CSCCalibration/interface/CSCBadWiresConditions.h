#ifndef _CSCBADWIRESCONDITIONS_H
#define _CSCBADWIRESCONDITIONS_H

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
#include "CondFormats/CSCObjects/interface/CSCBadWires.h"
#include "CondFormats/DataRecord/interface/CSCBadWiresRcd.h"

class CSCBadWiresConditions: public edm::ESProducer, public edm::EventSetupRecordIntervalFinder  {
 public:
  CSCBadWiresConditions(const edm::ParameterSet&);
  ~CSCBadWiresConditions();
  

  inline static CSCBadWires *  prefillBadWires();

  typedef const  CSCBadWires * ReturnType;
  
  ReturnType produceBadWires(const CSCBadWiresRcd&);
  
 private:
  // ----------member data ---------------------------
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&, edm::ValidityInterval & );
  CSCBadWires *cndbBadWires ;

};

#include<fstream>
#include<vector>
#include<iostream>

// to workaround plugin library
inline CSCBadWires *  CSCBadWiresConditions::prefillBadWires()
{
  //  const int MAX_SIZE = 217728;
  CSCBadWires * cndbbadwires = new CSCBadWires();
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
  newdata.open("badwires.dat",std::ios::in); 
  if(!newdata) {
    std::cerr <<"Error: badwires.dat -> no such file!"<< std::endl;
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
  

  CSCBadWires::BadChannelContainer & itemvector = cndbbadwires->channels;
  itemvector.resize(new_nrlines);
  CSCBadWires::BadChamberContainer & itemvector1 = cndbbadwires->chambers;
  itemvector1.resize(new_nrlines);
  cndbbadwires->numberOfBadChannels = new_nrlines;

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

  return cndbbadwires;
}

#endif

