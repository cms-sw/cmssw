#ifndef _CSCPEDESTALSDBCONDITIONS_H
#define _CSCPEDESTALSDBCONDITIONS_H

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <cmath>
#include <memory>

#include "CondFormats/CSCObjects/interface/CSCDBPedestals.h"
#include "CondFormats/DataRecord/interface/CSCDBPedestalsRcd.h"
#include <DataFormats/MuonDetId/interface/CSCDetId.h>

class CSCPedestalsDBConditions : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  CSCPedestalsDBConditions(const edm::ParameterSet &);
  ~CSCPedestalsDBConditions() override;

  inline static CSCDBPedestals *prefillDBPedestals();

  typedef std::unique_ptr<CSCDBPedestals> ReturnType;

  ReturnType produceDBPedestals(const CSCDBPedestalsRcd &);

private:
  // ----------member data ---------------------------
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &,
                      const edm::IOVSyncValue &,
                      edm::ValidityInterval &) override;
  CSCDBPedestals *cndbPedestals;
};

#include "CondFormats/CSCObjects/interface/CSCDBPedestals.h"
#include "CondFormats/DataRecord/interface/CSCDBPedestalsRcd.h"
#include <DataFormats/MuonDetId/interface/CSCDetId.h>

#include <fstream>
#include <iostream>
#include <vector>

// to workaround plugin library
inline CSCDBPedestals *CSCPedestalsDBConditions::prefillDBPedestals() {
  const int PED_FACTOR = 10;
  const int RMS_FACTOR = 1000;
  const int MAX_SIZE = 252288;
  // const int MAX_SIZE = 273024; //for extra ME1a unganged case
  const int MAX_SHORT = 32767;
  CSCDBPedestals *cndbpedestals = new CSCDBPedestals();

  int db_index;
  float db_ped, db_rms;
  std::vector<int> db_index_id;
  std::vector<float> db_peds;
  std::vector<float> db_pedrms;
  int new_index;
  float new_ped, new_rms;
  std::vector<int> new_index_id;
  std::vector<float> new_peds;
  std::vector<float> new_pedrms;

  int counter;

  std::ifstream dbdata;
  dbdata.open("old_dbpeds.dat", std::ios::in);
  if (!dbdata) {
    std::cerr << "Error: old_dbpeds.dat -> no such file!" << std::endl;
    exit(1);
  }

  while (!dbdata.eof()) {
    dbdata >> db_index >> db_ped >> db_rms;
    db_index_id.push_back(db_index);
    db_peds.push_back(db_ped);
    db_pedrms.push_back(db_rms);
  }
  dbdata.close();

  std::ifstream newdata;
  newdata.open("peds.dat", std::ios::in);
  if (!newdata) {
    std::cerr << "Error: peds.dat -> no such file!" << std::endl;
    exit(1);
  }

  while (!newdata.eof()) {
    newdata >> new_index >> new_ped >> new_rms;
    new_index_id.push_back(new_index);
    new_peds.push_back(new_ped);
    new_pedrms.push_back(new_rms);
  }
  newdata.close();

  CSCDBPedestals::PedestalContainer &itemvector = cndbpedestals->pedestals;
  itemvector.resize(MAX_SIZE);
  cndbpedestals->factor_ped = int(PED_FACTOR);
  cndbpedestals->factor_rms = int(RMS_FACTOR);

  for (int i = 0; i < MAX_SIZE; ++i) {
    itemvector[i].ped = (short int)(db_peds[i] * PED_FACTOR + 0.5);
    itemvector[i].rms = (short int)(db_pedrms[i] * RMS_FACTOR + 0.5);
  }

  for (int i = 0; i < MAX_SIZE; ++i) {
    counter = db_index_id[i];
    itemvector[i] = itemvector[counter];
    itemvector[i].ped = int(db_peds[i]);
    itemvector[i].rms = int(db_pedrms[i]);

    for (unsigned int k = 0; k < new_index_id.size() - 1; k++) {
      if (counter == new_index_id[k]) {
        if ((short int)(fabs(new_peds[k] * PED_FACTOR + 0.5)) < MAX_SHORT)
          itemvector[counter].ped = int(new_peds[k] * PED_FACTOR + 0.5);
        if ((short int)(fabs(new_pedrms[k] * RMS_FACTOR + 0.5)) < MAX_SHORT)
          itemvector[counter].rms = int(new_pedrms[k] * RMS_FACTOR + 0.5);
        itemvector[i] = itemvector[counter];
        // if (new_peds[k]==0) itemvector[counter].ped = int (db_peds[i]);
        // if (new_pedrms[k]==0) itemvector[counter].rms = int (db_pedrms[i]);
      }
    }

    if (counter > 223968) {
      itemvector[counter].ped = int(db_peds[i]);
      itemvector[counter].rms = int(db_pedrms[i]);
      itemvector[i] = itemvector[counter];
    }
  }
  return cndbpedestals;
}

#endif
