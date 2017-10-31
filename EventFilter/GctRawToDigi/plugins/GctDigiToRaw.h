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
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"

#include "EventFilter/GctRawToDigi/src/GctFormatTranslateMCLegacy.h"

//
// class decleration
//

class GctDigiToRaw : public edm::EDProducer {
 public:
  explicit GctDigiToRaw(const edm::ParameterSet&);
  ~GctDigiToRaw() override;
  
 private: // methods
  void beginJob() override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endJob() override ;
  
  void print(FEDRawData& data);

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

  // pack flags
  bool packRctEm_;
  bool packRctCalo_;

  // FED numbers
  int fedId_;            

  // print out for each event
  bool verbose_;

  // counter events
  int counter_;          
  
  // digi to block converter
  GctFormatTranslateMCLegacy formatTranslator_;

};

#endif
