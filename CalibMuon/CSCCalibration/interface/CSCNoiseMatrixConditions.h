#ifndef _CSCNOISEMATRIXCONDITIONS_H
#define _CSCNOISEMATRIXCONDITIONS_H

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <memory>

#include "CondFormats/CSCObjects/interface/CSCNoiseMatrix.h"
#include "CondFormats/DataRecord/interface/CSCNoiseMatrixRcd.h"
#include <DataFormats/MuonDetId/interface/CSCDetId.h>

class CSCNoiseMatrixConditions : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  CSCNoiseMatrixConditions(const edm::ParameterSet &);
  ~CSCNoiseMatrixConditions() override;

  static CSCNoiseMatrix *prefillNoiseMatrix();

  typedef std::unique_ptr<CSCNoiseMatrix> ReturnType;

  ReturnType produceNoiseMatrix(const CSCNoiseMatrixRcd &);

private:
  // ----------member data ---------------------------
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &,
                      const edm::IOVSyncValue &,
                      edm::ValidityInterval &) override;
  CSCNoiseMatrix *cnMatrix;
};

#endif
