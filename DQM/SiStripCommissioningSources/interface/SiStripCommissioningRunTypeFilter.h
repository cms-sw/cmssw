#ifndef _dqm_sistripcommissioningsources_SiStripCommissioningRunTypeFilter_h_
#define _dqm_sistripcommissioningsources_SiStripCommissioningRunTypeFilter_h_

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

#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include <FWCore/Utilities/interface/InputTag.h>
#include <DataFormats/SiStripCommon/interface/SiStripEventSummary.h>


//
// class declaration
//
class SiStripEventSummary;

class SiStripCommissioningRunTypeFilter : public edm::EDFilter {

   public:

      explicit SiStripCommissioningRunTypeFilter(const edm::ParameterSet&);
      ~SiStripCommissioningRunTypeFilter() override {}

   private:

      bool filter(edm::Event&, const edm::EventSetup&) override;
      
      // ----------member data ---------------------------
      //      edm::InputTag inputModuleLabel_;
      edm::EDGetTokenT<SiStripEventSummary> summaryToken_;
      std::vector<sistrip::RunType> runTypes_;

};

#endif
