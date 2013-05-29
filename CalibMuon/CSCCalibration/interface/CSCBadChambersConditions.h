#ifndef _CSCBADCHAMBERSCONDITIONS_H
#define _CSCBADCHAMBERSCONDITIONS_H

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
#include "CondFormats/CSCObjects/interface/CSCBadChambers.h"
#include "CondFormats/DataRecord/interface/CSCBadChambersRcd.h"

class CSCBadChambersConditions: public edm::ESProducer, public edm::EventSetupRecordIntervalFinder  {
 public:
  CSCBadChambersConditions(const edm::ParameterSet&);
  ~CSCBadChambersConditions();
  

  inline static CSCBadChambers *  prefillBadChambers();

  typedef const  CSCBadChambers * ReturnType;
  
  ReturnType produceBadChambers(const CSCBadChambersRcd&);
  
 private:
  // ----------member data ---------------------------
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&, edm::ValidityInterval & );
  CSCBadChambers *cndbBadChambers ;

};

#include<fstream>
#include<vector>
#include<iostream>

// to workaround plugin library
inline CSCBadChambers *  CSCBadChambersConditions::prefillBadChambers()
{
  //  const int MAX_SIZE = 468;
  //cndbbadchambers = new CSCBadChambers();

  int new_chambers;
  std::vector<int> new_badchambers;

  int new_nrlines;
  new_nrlines=0;
 
  std::ifstream newdata;

  newdata.open("badchambers.dat",std::ios::in);
  if(!newdata) {
    std::cerr <<"Error: badchambers.dat -> no such file!"<< std::endl;
    exit(1);
  }

  while(!newdata.eof() ) {
    newdata >> new_chambers;
    new_badchambers.push_back(new_chambers);
    ++new_nrlines;
  }
  newdata.close();

  CSCBadChambers * cndbbadchambers = new CSCBadChambers(new_nrlines, new_badchambers);  

  //std::cout <<"numberOfBadChambers "<<new_nrlines<<std::endl;

   return cndbbadchambers;
}

#endif
