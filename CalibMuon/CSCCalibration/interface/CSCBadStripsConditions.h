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
  //bool bad=false; //use boolean to fill bad channels

  int new_index, new_chan;
  int new_layer,new_channel, new_flag1, new_flag2, new_flag3,new_pointer;
  std::vector<int> new_index_id;
  std::vector<short int> new_layer_id;
  std::vector<short int> new_chan_id;
  std::vector<int> new_badchannels;
  std::vector<short int> new_flag1_id;
  std::vector<short int> new_flag2_id;
  std::vector<short int> new_flag3_id;
  std::vector<int> new_cham_id;
  std::vector<int> new_point;

  // int counter;
  //  int db_nrlines=0;
  int new_nrlines1=0;
  int new_nrlines2=0;
 
  std::ifstream newdata1;
  std::ifstream newdata2;

  newdata1.open("badstrips1.dat",std::ios::in); //chambercontainer
  if(!newdata1) {
    std::cerr <<"Error: badstrips1.dat -> no such file!"<< std::endl;
    exit(1);
  }
  newdata2.open("badstrips2.dat",std::ios::in); //channelcontainer
  if(!newdata2) {
    std::cerr <<"Error: badstrips2.dat -> no such file!"<< std::endl;
    exit(1);
  }

  while(!newdata1.eof() ) {
    newdata1 >> new_index >> new_pointer >> new_chan;
    new_index_id.push_back(new_index);
    new_point.push_back(new_pointer);
    new_badchannels.push_back(new_chan);
    new_nrlines1++;
  }
  newdata1.close();

  while (!newdata2.eof() ) { 
    newdata2 >> new_layer >> new_channel >> new_flag1 >> new_flag2 >> new_flag3; 
    new_layer_id.push_back(new_layer);
    new_chan_id.push_back(new_channel);
    new_flag1_id.push_back(new_flag1);
    new_flag2_id.push_back(new_flag2);
    new_flag3_id.push_back(new_flag3);
    new_nrlines2++;
  }
  newdata2.close();
  

  CSCBadStrips::BadChamberContainer & itemvector1 = cndbbadstrips->chambers;
  itemvector1.resize(new_nrlines1);
  
  CSCBadStrips::BadChannelContainer & itemvector2 = cndbbadstrips->channels;
  itemvector2.resize(new_nrlines2);
  
  // std::cout<<new_nrlines1<<" "<<new_nrlines2<<std::endl;
  cndbbadstrips->numberOfBadChannels = new_nrlines2;

  //if(new_index_id.empty()) bad=false; 
  //if(!new_index_id.empty()) bad=true;

  for(int i=0; i<new_nrlines1-1;i++){
    itemvector1[i].chamber_index = new_index_id[i];
    itemvector1[i].pointer = new_point[i];
    itemvector1[i].bad_channels = new_badchannels[i];
  }

  for(int j=0;j<new_nrlines2-1;j++) {
    itemvector2[j].layer =  new_layer_id[j];
    itemvector2[j].channel = new_chan_id[j];
    itemvector2[j].flag1 =  new_flag1_id[j];
    itemvector2[j].flag2 = new_flag2_id[j];
    itemvector2[j].flag3 = new_flag3_id[j];
  }

   return cndbbadstrips;
}

#endif

