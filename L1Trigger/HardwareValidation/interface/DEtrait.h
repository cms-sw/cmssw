#ifndef DE_TRAIT_H
#define DE_TRAIT_H

/*\class template DEtrait
 *\description data|emulation auxiliary template
               type associator trait struct 
 *\author Nuno Leonardo (CERN)
 *\date 07.03
 */

// L1 dataformats includes
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTTrackContainer.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/L1CSCTrackFinder/interface/L1CSCTrackCollection.h"
#include "DataFormats/L1CSCTrackFinder/interface/L1CSCStatusDigiCollection.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTTrackCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include "DataFormats/LTCDigi/interface/LTCDigi.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerEvmReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"

//d|e record
#include "DataFormats/L1Trigger/interface/L1DataEmulRecord.h"

namespace dedefs {

  const int DEnsys = 12; 

  enum SysList {
    ETP, HTP, RCT, GCT, DTP, DTF, 
    CTP, CTF, RPC, LTC, GMT, GLT
  };

  const std::string SystLabel[DEnsys] = {
    "ETP", "HTP", "RCT", "GCT", "DTP", "DTF", 
    "CTP", "CTF", "RPC", "LTC", "GMT", "GLT"
  };

  const std::string SystLabelExt[DEnsys] = {
    "ECAL", "HCAL", "RCT", "GCT", "DTTPG", "DTTF", 
    "CSCTPG", "CSCTF", "RPC", "LTC", "GMT", "GT"
  };

  enum ColList {
    ECALtp, HCALtp, 
    RCTem, RCTrgn, 
    GCTem, GCTjet, GCTisolaem, GCTnoisoem, GCTcenjets, GCTforjets, GCTtaujets, 
    GCTethad, GCTetmiss, GCTettot, GCThtmiss, GCThfring, GCThfbit, GCTisotaujets,
    GCTjetcnt, 
    DTtpPh, DTtpTh, DTtf, DTtftrk, 
    CSCtpa, CSCtpc, CSCtpl, CSCsta, CSCtf, CSCtftrk, CSCtftrc, CSCtfsta, 
    MUrtf,     RPCcen, RPCfor, 
    LTCi, 
    GMTmain, GMTcnd, GMTrdt, 
    GTdword
  };

}

typedef std::vector<L1MuDTChambPhDigi>    L1MuDTChambPhDigiCollection; 
typedef std::vector<L1MuDTChambThDigi>    L1MuDTChambThDigiCollection; 
typedef std::vector<L1MuRegionalCand>     L1MuRegionalCandCollection;
typedef std::vector<CSCALCTDigi>          CSCALCTDigiCollection_;
typedef std::vector<CSCCLCTDigi>          CSCCLCTDigiCollection_;
typedef std::vector<CSCCorrelatedLCTDigi> CSCCorrelatedLCTDigiCollection_;
typedef std::vector<L1CSCSPStatusDigi>    L1CSCSPStatusDigiCollection_;
typedef std::vector<L1MuGMTCand>          L1MuGMTCandCollection;
typedef std::vector<L1MuGMTReadoutRecord> L1MuGMTReadoutRecordCollection;
typedef std::vector<L1DataEmulDigi>       L1DEDigiCollection;

template <typename T> 
struct DEtrait {};

template<> 
struct DEtrait<EcalTrigPrimDigiCollection> {
  typedef EcalTrigPrimDigiCollection coll_type;
  typedef EcalTriggerPrimitiveDigi   cand_type;
  static inline int de_type() {return dedefs::ECALtp;}
};

template<> 
struct DEtrait<HcalTrigPrimDigiCollection> {
  typedef HcalTrigPrimDigiCollection coll_type;
  typedef HcalTriggerPrimitiveDigi   cand_type;
  static inline int de_type() {return dedefs::HCALtp;}
};

template<> 
struct DEtrait<L1CaloEmCollection> {
  typedef L1CaloEmCollection     coll_type;
  typedef L1CaloEmCand           cand_type;
  static inline int de_type() {return dedefs::RCTem;}
};

template<> 
struct DEtrait<L1CaloRegionCollection> {
  typedef L1CaloRegionCollection coll_type;
  typedef L1CaloRegion           cand_type;
  static inline int de_type() {return dedefs::RCTrgn;}
};

template<> 
struct DEtrait<L1GctEmCandCollection> {
  typedef L1GctEmCandCollection  coll_type;
  typedef L1GctEmCand            cand_type;
  static inline int de_type() {return dedefs::GCTem;}
};

template<> 
struct DEtrait<L1GctJetCandCollection> {
  typedef L1GctJetCandCollection coll_type;
  typedef L1GctJetCand           cand_type;
  static inline int de_type() {return dedefs::GCTjet;}
};

template<> 
struct DEtrait<L1GctEtHadCollection> {
  typedef L1GctEtHadCollection coll_type;
  typedef L1GctEtHad           cand_type;
  static inline int de_type() {return dedefs::GCTethad;}
};
template<> 
struct DEtrait<L1GctEtMissCollection> {
  typedef L1GctEtMissCollection coll_type;
  typedef L1GctEtMiss cand_type;
  static inline int de_type() {return dedefs::GCTetmiss;}
};
template<> 
struct DEtrait<L1GctEtTotalCollection> {
  typedef L1GctEtTotalCollection coll_type;
  typedef L1GctEtTotal           cand_type;
  static inline int de_type() {return dedefs::GCTettot;}
};
template<> 
struct DEtrait<L1GctHtMissCollection> {
  typedef L1GctHtMissCollection coll_type;
  typedef L1GctHtMiss           cand_type;
  static inline int de_type() {return dedefs::GCThtmiss;}
};
template<> 
struct DEtrait<L1GctHFRingEtSumsCollection> {
  typedef L1GctHFRingEtSumsCollection coll_type;
  typedef L1GctHFRingEtSums           cand_type;
  static inline int de_type() {return dedefs::GCThfring;}
};
template<> 
struct DEtrait<L1GctHFBitCountsCollection> {
  typedef L1GctHFBitCountsCollection coll_type;
  typedef L1GctHFBitCounts           cand_type;
  static inline int de_type() {return dedefs::GCThfbit;}
};
template<> 
struct DEtrait<L1GctJetCountsCollection> {
  typedef L1GctJetCountsCollection coll_type;
  typedef L1GctJetCounts           cand_type;
  static inline int de_type() {return dedefs::GCTjetcnt;}
};

template<> 
struct DEtrait<L1MuDTChambPhDigiCollection> {
  typedef L1MuDTChambPhDigiCollection coll_type;
  typedef L1MuDTChambPhDigi           cand_type;
  static inline int de_type() {return dedefs::DTtpPh;}
};
template<> 
struct DEtrait<L1MuDTChambThDigiCollection> {
  typedef L1MuDTChambThDigiCollection coll_type;
  typedef L1MuDTChambThDigi           cand_type;
  static inline int de_type() {return dedefs::DTtpTh;}
};

template<> 
struct DEtrait<L1MuRegionalCandCollection> {
  typedef L1MuRegionalCandCollection coll_type;
  typedef L1MuRegionalCand           cand_type;
  static inline int de_type() {return dedefs::MUrtf;}
};

template<> 
struct DEtrait<CSCALCTDigiCollection_> {
  typedef CSCALCTDigiCollection_ coll_type;
  typedef CSCALCTDigi            cand_type;
  static inline int de_type() {return dedefs::CSCtpa;}
};
template<> 
struct DEtrait<CSCCLCTDigiCollection_> {
  typedef CSCCLCTDigiCollection_ coll_type;
  typedef CSCCLCTDigi            cand_type;
  static inline int de_type() {return dedefs::CSCtpc;}
};

template<> 
struct DEtrait<CSCCorrelatedLCTDigiCollection_> {
  typedef CSCCorrelatedLCTDigiCollection_ coll_type;
  typedef CSCCorrelatedLCTDigi            cand_type;
  static inline int de_type() {return dedefs::CSCtpl;}
};

template<> 
struct DEtrait<L1CSCSPStatusDigiCollection_> {
  typedef L1CSCSPStatusDigiCollection_ coll_type;
  typedef L1CSCSPStatusDigi            cand_type;
  static inline int de_type() {return dedefs::CSCsta;}
};

template<> 
struct DEtrait<LTCDigiCollection> {
  typedef LTCDigiCollection coll_type;
  typedef LTCDigi           cand_type;
  static inline int de_type() {return dedefs::LTCi;}
};

template<> 
struct DEtrait<L1MuGMTCandCollection> {
  typedef L1MuGMTCandCollection coll_type;
  typedef L1MuGMTCand           cand_type;
  static inline int de_type() {return dedefs::GMTcnd;}
};

template<> 
struct DEtrait<L1MuGMTReadoutRecordCollection> {
  typedef L1MuGMTReadoutRecordCollection coll_type;
  typedef L1MuGMTReadoutRecord           cand_type;
  static inline int de_type() {return dedefs::GMTrdt;}
};

template<> 
struct DEtrait<DecisionWord> {
  typedef DecisionWord coll_type;
  typedef bool         cand_type;
  static inline int de_type() {return dedefs::GTdword;}
};

#endif
