#ifndef CSCViewDigi_H
#define CSCViewDigi_H

/**\class CSCViewDigi CSCViewDigi.h 

Location: EventFilter/CSCRawToDigi/interface/CSCViewDigi.h

*/

// Original Author:  Alexandre Sakharov
//         Created:  Sun May 10 15:43:28 CEST 2009

#include <memory>

#include <FWCore/Framework/interface/ConsumesCollector.h>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "EventFilter/CSCRawToDigi/interface/CSCALCTHeader.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCComparatorDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCRPCDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCDCCFormatStatusDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCDDUStatusDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCDCCStatusDigiCollection.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


class CSCViewDigi : public edm::EDAnalyzer {
   public:
      explicit CSCViewDigi(const edm::ParameterSet&);
      ~CSCViewDigi() override;


   private:

      void analyze(const edm::Event&, const edm::EventSetup&) override;
      void endJob() override ;

      bool WiresDigiDump, AlctDigiDump, ClctDigiDump, CorrClctDigiDump;
      bool StripDigiDump, ComparatorDigiDump, RpcDigiDump, StatusDigiDump;
      bool DDUStatusDigiDump, DCCStatusDigiDump;

      edm::EDGetTokenT<CSCWireDigiCollection>             wd_token;
      edm::EDGetTokenT<CSCStripDigiCollection>            sd_token;
      edm::EDGetTokenT<CSCComparatorDigiCollection>       cd_token;
      edm::EDGetTokenT<CSCRPCDigiCollection>              rd_token;
      edm::EDGetTokenT<CSCALCTDigiCollection>             al_token;
      edm::EDGetTokenT<CSCCLCTDigiCollection>             cl_token;
      edm::EDGetTokenT<CSCCorrelatedLCTDigiCollection>    co_token;
      edm::EDGetTokenT<CSCDCCFormatStatusDigiCollection>  st_token;
      edm::EDGetTokenT<CSCDDUStatusDigiCollection>        dd_token;
      edm::EDGetTokenT<CSCDCCStatusDigiCollection>        dc_token;

};

#endif

