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
//#include "DataFormats/EcalDigi/interface/EcalTriggerPrimitiveSample.h"
//#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"

//hcal tpg
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

//rct
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
//#include "DataFormats/L1CaloTrigger/interface/L1CaloRegionDetId.h"

//gct
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"

//dttf, csctf, rpctf
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"

//dttpg
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhDigi.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThDigi.h"

//csctpg
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"

//rpctpg

//gmt
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTExtendedCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"

//ltc
#include "DataFormats/LTCDigi/interface/LTCDigi.h"


//gt
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
//#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerSetup.h"
//#include "L1Trigger/GlobalTrigger/interface/L1GlobalTrigger.h"

//typedef std::vector<bool> DecisionWord; 


#include "L1Trigger/HardwareValidation/interface/DEcompare.h"

enum compareMode {ETP=0, HTP, RCT, GCT, DTT, RPC, LTC, iGMT, GT};

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
  std::string ETP_data_Label_;
  std::string ETP_emul_Label_;
  std::string HTP_data_Label_;
  std::string HTP_emul_Label_;
  std::string RCT_data_Label_;
  std::string RCT_emul_Label_;
  std::string GCT_data_Label_;
  std::string GCT_emul_Label_;
  /*
  std::string DTP_data_Label_;
  std::string DTP_emul_Label_;
  std::string DTF_data_Label_;
  std::string DTF_emul_Label_;
  std::string CTP_data_Label_;
  std::string CTP_emul_Label_;
  std::string CTF_data_Label_;
  std::string CTF_emul_Label_;
  std::string RTP_data_Label_;
  std::string RTP_emul_Label_;
  std::string RTF_data_Label_;
  std::string RTF_emul_Label_;
  std::string LTC_data_Label_;
  std::string LTC_emul_Label_;
  std::string GMT_data_Label_;
  std::string GMT_emul_Label_;
  */
  std::string GT_data_Label_;
  std::string GT_emul_Label_;

  bool doEtp_;
  bool doHtp_;
  bool doRct_;
  bool doGct_;
  /*
  bool doDtp_;
  bool doDtf_;
  bool doCtp_;
  bool doCtf_;
  bool doRtp_;
  bool doRtf_;
  bool doLtc_;
  bool doGmt_;
  */
  bool doGt_;

  bool etp_match;
  bool htp_match;
  bool rct_match;
  bool gct_match;
  /*
  bool dtp_match;
  bool dtf_match;
  bool ctp_match;
  bool ctf_match;
  bool rtp_match;
  bool rtf_match;
  bool ltc_match;
  bool gmt_match;
  */
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
