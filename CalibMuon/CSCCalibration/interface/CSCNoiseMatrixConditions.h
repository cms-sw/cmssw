#ifndef _CSCNOISEMATRIXCONDITIONS_H
#define _CSCNOISEMATRIXCONDITIONS_H

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
#include "CondFormats/CSCObjects/interface/CSCNoiseMatrix.h"
#include "CondFormats/DataRecord/interface/CSCNoiseMatrixRcd.h"

class CSCNoiseMatrixConditions: public edm::ESProducer, public edm::EventSetupRecordIntervalFinder  {
 public:
  CSCNoiseMatrixConditions(const edm::ParameterSet&);
  ~CSCNoiseMatrixConditions();
  
  int old_chamber_id,old_strip,new_chamber_id,new_strip;
  float old_elm33,old_elm34, old_elm44, old_elm35, old_elm45, old_elm55;
  float old_elm46, old_elm56, old_elm66, old_elm57, old_elm67, old_elm77;
  std::vector<int> old_cham_id;
  std::vector<int> old_strips;
  std::vector<float> old_elem33;
  std::vector<float> old_elem34;
  std::vector<float> old_elem44;
  std::vector<float> old_elem45;
  std::vector<float> old_elem35;
  std::vector<float> old_elem55;
  std::vector<float> old_elem46;
  std::vector<float> old_elem56;
  std::vector<float> old_elem66;
  std::vector<float> old_elem57;
  std::vector<float> old_elem67;
  std::vector<float> old_elem77;


  float new_elm33,new_elm34, new_elm44, new_elm35, new_elm45, new_elm55;
  float  new_elm46, new_elm56, new_elm66, new_elm57, new_elm67, new_elm77;
  std::vector<int> new_cham_id;
  std::vector<int> new_strips;
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

  void prefillNoiseMatrix();
  

      typedef const  CSCNoiseMatrix * ReturnType;

      ReturnType produceNoiseMatrix(const CSCNoiseMatrixRcd&);

   private:
      // ----------member data ---------------------------
    void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&, edm::ValidityInterval & );
    CSCNoiseMatrix *cnmatrix ;

};

#endif
