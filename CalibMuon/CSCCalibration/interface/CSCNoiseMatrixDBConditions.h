#ifndef _CSCNOISEMATRIXDBCONDITIONS_H
#define _CSCNOISEMATRIXDBCONDITIONS_H

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

#include "CondFormats/CSCObjects/interface/CSCDBNoiseMatrix.h"
#include "CondFormats/DataRecord/interface/CSCDBNoiseMatrixRcd.h"
#include <DataFormats/MuonDetId/interface/CSCDetId.h>

class CSCNoiseMatrixDBConditions : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  CSCNoiseMatrixDBConditions(const edm::ParameterSet &);
  ~CSCNoiseMatrixDBConditions() override;

  inline static CSCDBNoiseMatrix *prefillDBNoiseMatrix();

  typedef std::unique_ptr<CSCDBNoiseMatrix> ReturnType;

  ReturnType produceDBNoiseMatrix(const CSCDBNoiseMatrixRcd &);

private:
  // ----------member data ---------------------------
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &,
                      const edm::IOVSyncValue &,
                      edm::ValidityInterval &) override;
  CSCDBNoiseMatrix *cndbMatrix;
};

#include <fstream>
#include <iostream>
#include <vector>

// to workaround plugin library
inline CSCDBNoiseMatrix *CSCNoiseMatrixDBConditions::prefillDBNoiseMatrix() {
  // const int MAX_SIZE = 273024; //for ME1a unganged
  const int MAX_SIZE = 252288;
  const int FACTOR = 1000;
  const int MAX_SHORT = 32767;

  int new_index, db_index;
  float db_elm33, db_elm34, db_elm44, db_elm35, db_elm45, db_elm55;
  float db_elm46, db_elm56, db_elm66, db_elm57, db_elm67, db_elm77;
  std::vector<int> db_index_id;
  std::vector<float> db_elem33;
  std::vector<float> db_elem34;
  std::vector<float> db_elem44;
  std::vector<float> db_elem45;
  std::vector<float> db_elem35;
  std::vector<float> db_elem55;
  std::vector<float> db_elem46;
  std::vector<float> db_elem56;
  std::vector<float> db_elem66;
  std::vector<float> db_elem57;
  std::vector<float> db_elem67;
  std::vector<float> db_elem77;

  float new_elm33, new_elm34, new_elm44, new_elm35, new_elm45, new_elm55;
  float new_elm46, new_elm56, new_elm66, new_elm57, new_elm67, new_elm77;
  std::vector<int> new_cham_id;
  std::vector<int> new_index_id;
  std::vector<float> new_elem33;
  std::vector<float> new_elem34;
  std::vector<float> new_elem44;
  std::vector<float> new_elem45;
  std::vector<float> new_elem35;
  std::vector<float> new_elem55;
  std::vector<float> new_elem46;
  std::vector<float> new_elem56;
  std::vector<float> new_elem66;
  std::vector<float> new_elem57;
  std::vector<float> new_elem67;
  std::vector<float> new_elem77;

  CSCDBNoiseMatrix *cndbmatrix = new CSCDBNoiseMatrix();

  int counter;

  std::ifstream dbdata;
  dbdata.open("old_dbmatrix.dat", std::ios::in);
  if (!dbdata) {
    std::cerr << "Error: old_dbmatrix.dat -> no such file!" << std::endl;
    exit(1);
  }

  while (!dbdata.eof()) {
    dbdata >> db_index >> db_elm33 >> db_elm34 >> db_elm44 >> db_elm35 >> db_elm45 >> db_elm55 >> db_elm46 >>
        db_elm56 >> db_elm66 >> db_elm57 >> db_elm67 >> db_elm77;
    db_index_id.push_back(db_index);
    db_elem33.push_back(db_elm33);
    db_elem34.push_back(db_elm34);
    db_elem35.push_back(db_elm35);
    db_elem44.push_back(db_elm44);
    db_elem45.push_back(db_elm45);
    db_elem46.push_back(db_elm46);
    db_elem55.push_back(db_elm55);
    db_elem56.push_back(db_elm56);
    db_elem57.push_back(db_elm57);
    db_elem66.push_back(db_elm66);
    db_elem67.push_back(db_elm67);
    db_elem77.push_back(db_elm77);
  }
  dbdata.close();

  std::ifstream newdata;
  newdata.open("matrix.dat", std::ios::in);
  if (!newdata) {
    std::cerr << "Error: matrix.dat -> no such file!" << std::endl;
    exit(1);
  }

  while (!newdata.eof()) {
    newdata >> new_index >> new_elm33 >> new_elm34 >> new_elm44 >> new_elm35 >> new_elm45 >> new_elm55 >> new_elm46 >>
        new_elm56 >> new_elm66 >> new_elm57 >> new_elm67 >> new_elm77;
    // new_cham_id.push_back(new_chamber_id);
    new_index_id.push_back(new_index);
    new_elem33.push_back(new_elm33);
    new_elem34.push_back(new_elm34);
    new_elem35.push_back(new_elm35);
    new_elem44.push_back(new_elm44);
    new_elem45.push_back(new_elm45);
    new_elem46.push_back(new_elm46);
    new_elem55.push_back(new_elm55);
    new_elem56.push_back(new_elm56);
    new_elem57.push_back(new_elm57);
    new_elem66.push_back(new_elm66);
    new_elem67.push_back(new_elm67);
    new_elem77.push_back(new_elm77);
  }
  newdata.close();

  CSCDBNoiseMatrix::NoiseMatrixContainer &itemvector = cndbmatrix->matrix;
  itemvector.resize(MAX_SIZE);
  cndbmatrix->factor_noise = int(FACTOR);

  for (int i = 0; i < MAX_SIZE; ++i) {
    itemvector[i].elem33 = (short int)(db_elem33[i] * FACTOR + 0.5);
    itemvector[i].elem34 = (short int)(db_elem34[i] * FACTOR + 0.5);
    itemvector[i].elem35 = (short int)(db_elem35[i] * FACTOR + 0.5);
    itemvector[i].elem44 = (short int)(db_elem44[i] * FACTOR + 0.5);
    itemvector[i].elem45 = (short int)(db_elem45[i] * FACTOR + 0.5);
    itemvector[i].elem46 = (short int)(db_elem46[i] * FACTOR + 0.5);
    itemvector[i].elem55 = (short int)(db_elem55[i] * FACTOR + 0.5);
    itemvector[i].elem56 = (short int)(db_elem56[i] * FACTOR + 0.5);
    itemvector[i].elem57 = (short int)(db_elem57[i] * FACTOR + 0.5);
    itemvector[i].elem66 = (short int)(db_elem66[i] * FACTOR + 0.5);
    itemvector[i].elem67 = (short int)(db_elem67[i] * FACTOR + 0.5);
    itemvector[i].elem77 = (short int)(db_elem77[i] * FACTOR + 0.5);
  }

  for (int i = 0; i < MAX_SIZE; ++i) {
    counter = db_index_id[i];
    itemvector[i] = itemvector[counter];
    itemvector[i].elem33 = int(db_elem33[i]);
    itemvector[i].elem34 = int(db_elem34[i]);
    itemvector[i].elem35 = int(db_elem35[i]);
    itemvector[i].elem44 = int(db_elem44[i]);
    itemvector[i].elem45 = int(db_elem45[i]);
    itemvector[i].elem46 = int(db_elem46[i]);
    itemvector[i].elem55 = int(db_elem55[i]);
    itemvector[i].elem56 = int(db_elem56[i]);
    itemvector[i].elem57 = int(db_elem57[i]);
    itemvector[i].elem66 = int(db_elem66[i]);
    itemvector[i].elem67 = int(db_elem67[i]);
    itemvector[i].elem77 = int(db_elem77[i]);

    for (unsigned int k = 0; k < new_index_id.size() - 1; k++) {
      if (counter == new_index_id[k]) {
        if ((short int)(fabs(new_elem33[k] * FACTOR + 0.5)) < MAX_SHORT)
          itemvector[counter].elem33 = int(new_elem33[k] * FACTOR + 0.5);
        if ((short int)(fabs(new_elem34[k] * FACTOR + 0.5)) < MAX_SHORT)
          itemvector[counter].elem34 = int(new_elem34[k] * FACTOR + 0.5);
        if ((short int)(fabs(new_elem35[k] * FACTOR + 0.5)) < MAX_SHORT)
          itemvector[counter].elem35 = int(new_elem35[k] * FACTOR + 0.5);
        if ((short int)(fabs(new_elem44[k] * FACTOR + 0.5)) < MAX_SHORT)
          itemvector[counter].elem44 = int(new_elem44[k] * FACTOR + 0.5);
        if ((short int)(fabs(new_elem45[k] * FACTOR + 0.5)) < MAX_SHORT)
          itemvector[counter].elem45 = int(new_elem45[k] * FACTOR + 0.5);
        if ((short int)(fabs(new_elem46[k] * FACTOR + 0.5)) < MAX_SHORT)
          itemvector[counter].elem46 = int(new_elem46[k] * FACTOR + 0.5);
        if ((short int)(fabs(new_elem55[k] * FACTOR + 0.5)) < MAX_SHORT)
          itemvector[counter].elem55 = int(new_elem55[k] * FACTOR + 0.5);
        if ((short int)(fabs(new_elem56[k] * FACTOR + 0.5)) < MAX_SHORT)
          itemvector[counter].elem56 = int(new_elem56[k] * FACTOR + 0.5);
        if ((short int)(fabs(new_elem57[k] * FACTOR + 0.5)) < MAX_SHORT)
          itemvector[counter].elem57 = int(new_elem57[k] * FACTOR + 0.5);
        if ((short int)(fabs(new_elem66[k] * FACTOR + 0.5)) < MAX_SHORT)
          itemvector[counter].elem66 = int(new_elem66[k] * FACTOR + 0.5);
        if ((short int)(fabs(new_elem67[k] * FACTOR + 0.5)) < MAX_SHORT)
          itemvector[counter].elem67 = int(new_elem67[k] * FACTOR + 0.5);
        if ((short int)(fabs(new_elem77[k] * FACTOR + 0.5)) < MAX_SHORT)
          itemvector[counter].elem77 = int(new_elem77[k] * FACTOR + 0.5);
        itemvector[i] = itemvector[counter];
      }
    }

    if (counter > 223968) {
      itemvector[counter].elem33 = int(db_elem33[i]);
      itemvector[counter].elem34 = int(db_elem34[i]);
      itemvector[counter].elem35 = int(db_elem35[i]);
      itemvector[counter].elem44 = int(db_elem44[i]);
      itemvector[counter].elem45 = int(db_elem45[i]);
      itemvector[counter].elem46 = int(db_elem46[i]);
      itemvector[counter].elem55 = int(db_elem55[i]);
      itemvector[counter].elem56 = int(db_elem56[i]);
      itemvector[counter].elem57 = int(db_elem57[i]);
      itemvector[counter].elem66 = int(db_elem66[i]);
      itemvector[counter].elem67 = int(db_elem67[i]);
      itemvector[counter].elem77 = int(db_elem77[i]);
      itemvector[i] = itemvector[counter];
    }
  }

  return cndbmatrix;
}

#endif
