#ifndef L1COMPARATOR_H
#define L1COMPARATOR_H

/*\class L1Comparator
 *\description L1 trigger data|emulation comparison and validation
 *\author Nuno Leonardo (CERN)
 *\date 07.02
 */

// system include files
#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <algorithm>

// user include files
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//ecal tpg
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

//hcal tpg
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

//rct
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"

//gct
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"

//dtp
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"

//dtf
#include <DataFormats/L1DTTrackFinder/interface/L1MuDTTrackContainer.h>

//ctp
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"

//csctf
#include "DataFormats/L1CSCTrackFinder/interface/L1CSCTrackCollection.h"

//rpc,..
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"

//ltc
#include "DataFormats/LTCDigi/interface/LTCDigi.h"

//gmt
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"

//gt
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"


#include "L1Trigger/HardwareValidation/interface/DEcompare.h"

enum compareMode {ETP=0, HTP, RCT, GCT, DTP, DTF, CTP, CTF, RPC, LTC, GMTi, GT};

template <class T> class DEcompare;
 

class L1Comparator : public edm::EDAnalyzer {
public:
  explicit L1Comparator(const edm::ParameterSet&);
  ~L1Comparator();
  
private:
  virtual void beginJob(const edm::EventSetup&);
  virtual void analyze (const edm::Event&, const edm::EventSetup&);
  virtual void endJob();

  /// member data
  edm::InputTag ETP_data_Label_;
  edm::InputTag ETP_emul_Label_;
  edm::InputTag HTP_data_Label_;
  edm::InputTag HTP_emul_Label_;
  edm::InputTag RCT_data_Label_;
  edm::InputTag RCT_emul_Label_;
  edm::InputTag GCT_data_Label_;
  edm::InputTag GCT_emul_Label_;
  edm::InputTag DTP_data_Label_;
  edm::InputTag DTP_emul_Label_;
  edm::InputTag DTF_data_Label_;
  edm::InputTag DTF_emul_Label_;
  edm::InputTag CTP_data_Label_;
  edm::InputTag CTP_emul_Label_;
  edm::InputTag CTF_data_Label_;
  edm::InputTag CTF_emul_Label_;
  edm::InputTag RPC_data_Label_;
  edm::InputTag RPC_emul_Label_;
  edm::InputTag LTC_data_Label_;
  edm::InputTag LTC_emul_Label_;
  edm::InputTag GMT_data_Label_;
  edm::InputTag GMT_emul_Label_;
  edm::InputTag GT_data_Label_;
  edm::InputTag GT_emul_Label_;

  bool doEtp_;
  bool doHtp_;
  bool doRct_;
  bool doGct_;
  bool doDtp_;
  bool doDtf_;
  bool doCtp_;
  bool doCtf_;
  bool doRpc_;
  bool doLtc_;
  bool doGmt_;
  bool doGt_;

  bool etp_match;
  bool htp_match;
  bool rct_match;
  bool gct_match;
  bool dtp_match;
  bool dtf_match;
  bool ctp_match;
  bool ctf_match;
  bool rpc_match;
  bool ltc_match;
  bool gmt_match;
  bool gt_match;
  bool evt_match;
  bool all_match;

  std::string dumpFileName;
  std::ofstream dumpFile;
  int dumpMode;

  bool ReadCollections();

  ///DEBUG&alternatives
  bool dumpCandidate (L1CaloEmCand&, L1CaloEmCand&, std::ostream& s=std::cout);
  bool dumpCandidate (const L1GctEmCand&, const L1GctEmCand&, std::ostream& s=std::cout);
  bool compareCollections(edm::Handle<L1CaloEmCollection>    data, edm::Handle<L1CaloEmCollection>    emul);
  bool compareCollections(edm::Handle<L1GctEmCandCollection> data, edm::Handle<L1GctEmCandCollection> emul);
  bool compareCollections(edm::Handle<L1GlobalTriggerReadoutRecord> data, edm::Handle<L1GlobalTriggerReadoutRecord> emul);
  template <class T> bool CompareCollections( edm::Handle<T> data, edm::Handle<T> emul);
};

#endif
