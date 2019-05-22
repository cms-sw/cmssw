#ifndef _dqm_sistripcommissioningsources_SiStripCommissioningSeedFilter_h_
#define _dqm_sistripcommissioningsources_SiStripCommissioningSeedFilter_h_

// system include files
#include <memory>
#include <vector>

// user include files
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <FWCore/Utilities/interface/InputTag.h>

//
// class declaration
//
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

class SiStripCommissioningSeedFilter : public edm::EDFilter {
public:
  explicit SiStripCommissioningSeedFilter(const edm::ParameterSet&);
  ~SiStripCommissioningSeedFilter() override {}

private:
  bool filter(edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------
  //      edm::InputTag inputModuleLabel_;
  edm::EDGetTokenT<TrajectorySeedCollection> seedcollToken_;
};

#endif
