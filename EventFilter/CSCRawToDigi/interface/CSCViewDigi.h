/// -*- C++ -*-
//
// Package:    CSCViewDigi
// Class:      CSCViewDigi
// 
/**\class CSCViewDigi CSCViewDigi.h 

Location: EventFilter/CSCRawToDigi/interface/ViewDigi.h

Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Alexandre Sakharov
//         Created:  Sun May 10 15:43:28 CEST 2009
// $Id: CSCViewDigi.h,v 1.5 2010/07/21 04:23:23 wmtan Exp $
//
//


// system include files
#include <memory>

// user include files
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
//#include <EventFilter/CSCRawToDigi/interface/CSCDCCUnpacker.h>
//#include <iostream> CSCCorrelatedLCTDigi

//
// class decleration
//

#ifndef SCViewDigi_H
#define SCViewDigi_H

class CSCViewDigi : public edm::EDAnalyzer {
   public:
      explicit CSCViewDigi(const edm::ParameterSet&);
      ~CSCViewDigi();


   private:
      edm::InputTag wireDigiTag_;
      edm::InputTag alctDigiTag_;
      edm::InputTag clctDigiTag_;
      edm::InputTag corrclctDigiTag_;
      edm::InputTag stripDigiTag_;
      edm::InputTag comparatorDigiTag_;
      edm::InputTag rpcDigiTag_;
      edm::InputTag statusDigiTag_;
      edm::InputTag DDUstatusDigiTag_;
      edm::InputTag DCCstatusDigiTag_;
      bool WiresDigiDump, AlctDigiDump, ClctDigiDump, CorrClctDigiDump;
      bool StripDigiDump, ComparatorDigiDump, RpcDigiDump, StatusDigiDump;
      bool DDUStatusDigiDump, DCCStatusDigiDump;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------
};

#endif

