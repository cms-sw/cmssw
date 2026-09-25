#ifndef _CSCCROSSTALKCONDITIONS_H
#define _CSCCROSSTALKCONDITIONS_H

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EventSetupRecordInfiniteIntervalFinder.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include <memory>

#include "CondFormats/CSCObjects/interface/CSCcrosstalk.h"
#include "CondFormats/DataRecord/interface/CSCcrosstalkRcd.h"
#include <DataFormats/MuonDetId/interface/CSCDetId.h>

class CSCCrosstalkConditions : public edm::ESProducer, public edm::EventSetupRecordInfiniteIntervalFinder {
public:
  explicit CSCCrosstalkConditions(const edm::ParameterSet &);
  ~CSCCrosstalkConditions() override;

  static CSCcrosstalk *prefillCrosstalk();

  typedef std::unique_ptr<CSCcrosstalk> ReturnType;

  ReturnType produceCrosstalk(const CSCcrosstalkRcd &);

private:
  // ----------member data ---------------------------
};

#endif
