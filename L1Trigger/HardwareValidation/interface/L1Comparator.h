#ifndef L1COMPARATOR_H
#define L1COMPARATOR_H

/*\class L1Comparator
 *\description L1 trigger data|emulation comparison and validation
 *\author Nuno Leonardo (CERN)
 *\date 07.02
 */

// common/system includes
#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

// l1 dataformats, d|e record includes
#include "L1Trigger/HardwareValidation/interface/DEtrait.h"

// comparator template
#include "L1Trigger/HardwareValidation/interface/DEcompare.h"

// db trigger/subsystem key access
#include "CondFormats/L1TObjects/interface/L1TriggerKey.h"	   
//#include "CondFormats/L1TObjects/interface/L1TriggerKeyList.h"   
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"	   
//#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"

template <class T> class DEcompare;

class L1Comparator : public edm::EDProducer {

public:

  explicit L1Comparator(const edm::ParameterSet&);
  ~L1Comparator();
  
private:

  virtual void beginJob(void);
  virtual void beginRun(edm::Run const&, const edm::EventSetup&) override final;
   virtual void produce (edm::Event&, const edm::EventSetup&);
  virtual void endJob();

  template <class T> 
    void process( T const*, T const*, const int, const int);
  template <class T> 
    void process(const edm::Handle<T> data, const edm::Handle<T> emul, 
		 const int sys, const int cid) {
    if(data.isValid()&&emul.isValid())
      process(data.product(),emul.product(),sys, cid);
  }

  template <class T> bool CompareCollections(edm::Handle<T> data, edm::Handle<T> emul);
  template <class T> bool dumpCandidate(const T& dt, const T& em, std::ostream& s);

  int verbose() {return verbose_;}
  bool m_stage1_layer2_;  

 private:

  int nevt_;
  int evtNum_;
  int runNum_;
  int verbose_;
  bool dumpEvent_;

  edm::EDGetTokenT<L1CaloEmCollection> tokenCaloEm_[2];
  edm::EDGetTokenT<L1CaloRegionCollection> tokenCaloRegion_[2];
  edm::EDGetTokenT<L1GctEmCandCollection> tokenGctEmCand_isoEm_[2];
  edm::EDGetTokenT<L1GctEmCandCollection> tokenGctEmCand_nonIsoEm_[2];
  edm::EDGetTokenT<L1GctJetCandCollection> tokenGctJetCand_cenJets_[2];
  edm::EDGetTokenT<L1GctJetCandCollection> tokenGctJetCand_forJets_[2];
  edm::EDGetTokenT<L1GctJetCandCollection> tokenGctJetCand_tauJets_[2];
  edm::EDGetTokenT<L1GctJetCandCollection> tokenGctJetCand_isoTauJets_[2];
  edm::EDGetTokenT<L1GctEtTotalCollection> tokenGctEtTotal_[2];
  edm::EDGetTokenT<L1GctEtHadCollection> tokenGctEtHad_[2];
  edm::EDGetTokenT<L1GctEtMissCollection> tokenGctEtMiss_[2];
  edm::EDGetTokenT<L1GctHFRingEtSumsCollection> tokenGctHFRingEtSums_[2];
  edm::EDGetTokenT<L1GctHFBitCountsCollection> tokenGctHFBitCounts_[2];
  edm::EDGetTokenT<L1GctHtMissCollection> tokenGctHtMiss_[2];
  edm::EDGetTokenT<L1GctJetCountsCollection> tokenGctJetCounts_[2];
  edm::EDGetTokenT<L1MuDTChambPhContainer> tokenMuDTChambPh_[2];
  edm::EDGetTokenT<L1MuDTChambThContainer> tokenMuDTChambTh_[2];
  edm::EDGetTokenT<LTCDigiCollection> tokenLTCDigi_[2];
  edm::EDGetTokenT<L1MuDTTrackContainer> tokenMuDTTrack_[2];
  edm::EDGetTokenT<L1MuRegionalCandCollection> tokenMuRegionalCandRPCb_[2];
  edm::EDGetTokenT<L1MuRegionalCandCollection> tokenMuRegionalCandRPCf_[2];
  edm::EDGetTokenT<L1MuGMTCandCollection> tokenMuGMTCand_[2];
  edm::EDGetTokenT<L1MuGMTReadoutCollection> tokenMuReadoutCand_[2];

  bool m_doSys[dedefs::DEnsys];
  std::string m_dumpFileName;
  std::ofstream m_dumpFile;
  int m_dumpMode;
  bool m_match;
  bool DEmatchEvt[dedefs::DEnsys]; 
  int DEncand[dedefs::DEnsys][2];
  L1DEDigiCollection m_dedigis;

};

#endif
