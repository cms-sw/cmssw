#ifndef _CSCFRONTIERCROSSTALKCONDITIONS_H
#define _CSCFRONTIERCROSSTALKCONDITIONS_H

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EventSetupRecordInfiniteIntervalFinder.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <memory>

#include "CondFormats/CSCObjects/interface/CSCcrosstalk.h"
#include "CondFormats/DataRecord/interface/CSCcrosstalkRcd.h"
#include <DataFormats/MuonDetId/interface/CSCDetId.h>

class CSCFakeCrosstalkConditions : public edm::ESProducer, public edm::EventSetupRecordInfiniteIntervalFinder {
public:
  explicit CSCFakeCrosstalkConditions(const edm::ParameterSet &);
  ~CSCFakeCrosstalkConditions() override;

  static CSCcrosstalk *prefillCrosstalk();

  typedef std::unique_ptr<CSCcrosstalk> ReturnType;

  ReturnType produceCrosstalk(const CSCcrosstalkRcd &);

private:
  // ----------member data ---------------------------
};

#endif
