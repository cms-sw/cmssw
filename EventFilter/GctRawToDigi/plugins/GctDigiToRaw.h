#ifndef GctDigiToRaw_h
#define GctDigiToRaw_h

// -*- C++ -*-
//
// Package:    GctDigiToRaw
// Class:      GctDigiToRaw
// 
/**\class GctDigiToRaw GctDigiToRaw.cc EventFilter/GctRawToDigi/src/GctDigiToRaw.cc

 Description: Produce fake GCT raw data from digis

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jim Brooke
//         Created:  Wed Nov  1 11:57:10 CET 2006
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"

#include "EventFilter/GctRawToDigi/src/GctFormatTranslateMCLegacy.h"

//
// class decleration
//

class GctDigiToRaw : public edm::global::EDProducer<> {
 public:
  explicit GctDigiToRaw(const edm::ParameterSet&);
  
 private: // methods
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const final;
  
  void print(FEDRawData& data) const;

 private:  // members

  // input tokens
  edm::EDGetTokenT<L1GctEmCandCollection> tokenL1GctEmCand_isoEm_;
  edm::EDGetTokenT<L1GctEmCandCollection> tokenL1GctEmCand_nonIsoEm_;
  edm::EDGetTokenT<L1GctJetCandCollection> tokenGctJetCand_cenJets_;
  edm::EDGetTokenT<L1GctJetCandCollection> tokenGctJetCand_forJets_;
  edm::EDGetTokenT<L1GctJetCandCollection> tokenGctJetCand_tauJets_;
  edm::EDGetTokenT<L1GctEtTotalCollection> tokenGctEtTotal_;
  edm::EDGetTokenT<L1GctEtHadCollection> tokenGctEtHad_;
  edm::EDGetTokenT<L1GctEtMissCollection> tokenGctEtMiss_;
  edm::EDGetTokenT<L1GctHFRingEtSumsCollection> tokenGctHFRingEtSums_;
  edm::EDGetTokenT<L1GctHFBitCountsCollection> tokenGctHFBitCounts_;
  edm::EDGetTokenT<L1GctHtMissCollection> tokenGctHtMiss_;
  edm::EDGetTokenT<L1GctJetCountsCollection> tokenGctJetCounts_;
  edm::EDGetTokenT<L1CaloEmCollection> tokenCaloEm_;
  edm::EDGetTokenT<L1CaloRegionCollection> tokenCaloRegion_;
  edm::EDPutTokenT<FEDRawDataCollection> tokenPut_;
  // pack flags
  const bool packRctEm_;
  const bool packRctCalo_;

  // FED numbers
  const int fedId_;            

  // print out for each event
  const bool verbose_;

  // counter events
  mutable std::atomic<int> counter_;          

};

#endif
