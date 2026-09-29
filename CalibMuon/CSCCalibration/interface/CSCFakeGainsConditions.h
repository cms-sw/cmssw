#ifndef _CSCFAKEGAINSCONDITIONS_H
#define _CSCFAKEGAINSCONDITIONS_H

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EventSetupRecordInfiniteIntervalFinder.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include <memory>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/CSCObjects/interface/CSCGains.h"
#include "CondFormats/DataRecord/interface/CSCGainsRcd.h"
#include <DataFormats/MuonDetId/interface/CSCDetId.h>

class CSCFakeGainsConditions : public edm::ESProducer, public edm::EventSetupRecordInfiniteIntervalFinder {
public:
  CSCFakeGainsConditions(const edm::ParameterSet &);
  ~CSCFakeGainsConditions() override;

  static CSCGains *prefillGains();

  typedef std::unique_ptr<CSCGains> ReturnType;

  ReturnType produceGains(const CSCGainsRcd &);

private:
  // ----------member data ---------------------------
};

#endif
