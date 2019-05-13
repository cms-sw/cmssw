#ifndef _CSCFAKENOISEMATRIXCONDITIONS_H
#define _CSCFAKENOISEMATRIXCONDITIONS_H

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

class CSCFakeNoiseMatrixConditions : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  CSCFakeNoiseMatrixConditions(const edm::ParameterSet &);
  ~CSCFakeNoiseMatrixConditions() override;

  CSCNoiseMatrix *prefillNoiseMatrix();

  typedef std::unique_ptr<CSCNoiseMatrix> ReturnType;
  ReturnType produceNoiseMatrix(const CSCNoiseMatrixRcd &);

private:
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &,
                      const edm::IOVSyncValue &,
                      edm::ValidityInterval &) override;
};

#endif
