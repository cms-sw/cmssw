#ifndef _CSCFAKEPEDESTALSCONDITIONS_H
#define _CSCFAKEPEDESTALSCONDITIONS_H

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

#include "CondFormats/CSCObjects/interface/CSCPedestals.h"
#include "CondFormats/DataRecord/interface/CSCPedestalsRcd.h"
#include <DataFormats/MuonDetId/interface/CSCDetId.h>

class CSCFakePedestalsConditions : public edm::ESProducer, public edm::EventSetupRecordInfiniteIntervalFinder {
public:
  explicit CSCFakePedestalsConditions(const edm::ParameterSet &);
  ~CSCFakePedestalsConditions() override;

  static CSCPedestals *prefillPedestals();

  typedef std::unique_ptr<CSCPedestals> ReturnType;

  ReturnType producePedestals(const CSCPedestalsRcd &);

private:
  // ----------member data ---------------------------
};

#endif
