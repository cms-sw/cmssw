#ifndef DETRAIT_H
#define DETRAIT_H

/*\class template DEtrait
 *\description data|emulation auxiliary template
               type associator trait struct 
 *\author Nuno Leonardo (CERN)
 *\date 07.03
 */


enum collSet {ECALtp=0, HCALtp, RCTem, RCTrgn, GCTem, GCTjet, DTtpPh, DTtpTh, CSCtp, CSCtf, RPCb, RPCf, LTCi, GMTcnd, GMTrdt, GTdword};

typedef std::vector<L1MuDTChambPhDigi>    L1MuDTChambPhDigiCollection; 
typedef std::vector<L1MuDTChambThDigi>    L1MuDTChambThDigiCollection; 
typedef std::vector<L1MuRegionalCand>     L1MuRegionalCandCollection;
typedef std::vector<CSCCorrelatedLCTDigi> CSCCorrelatedLCTDigiCollection_;
typedef std::vector<L1MuGMTCand>          L1MuGMTCandCollection;
typedef std::vector<L1MuGMTReadoutRecord> L1MuGMTReadoutRecordCollection;

template <typename T>
struct DEtrait {};


template<> 
struct DEtrait<EcalTrigPrimDigiCollection> {
  typedef EcalTrigPrimDigiCollection coll_type;
  typedef EcalTriggerPrimitiveDigi   cand_type;
  static inline int de_type() {return ECALtp;}
};

template<> 
struct DEtrait<HcalTrigPrimDigiCollection> {
  typedef HcalTrigPrimDigiCollection coll_type;
  typedef HcalTriggerPrimitiveDigi   cand_type;
  static inline int de_type() {return HCALtp;}
};

template<> 
struct DEtrait<L1CaloEmCollection> {
  typedef L1CaloEmCollection     coll_type;
  typedef L1CaloEmCand           cand_type;
  static inline int de_type() {return RCTem;}
};

template<> 
struct DEtrait<L1CaloRegionCollection> {
  typedef L1CaloRegionCollection coll_type;
  typedef L1CaloRegion           cand_type;
  static inline int de_type() {return RCTrgn;}
};

template<> 
struct DEtrait<L1GctEmCandCollection> {
  typedef L1GctEmCandCollection  coll_type;
  typedef L1GctEmCand            cand_type;
  static inline int de_type() {return GCTem;}
};

template<> 
struct DEtrait<L1GctJetCandCollection> {
  typedef L1GctJetCandCollection coll_type;
  typedef L1GctJetCand           cand_type;
  static inline int de_type() {return GCTjet;}
};

template<> 
struct DEtrait<L1MuDTChambPhDigiCollection> {
  typedef L1MuDTChambPhDigiCollection coll_type;
  typedef L1MuDTChambPhDigi           cand_type;
  static inline int de_type() {return DTtpPh;}
};
template<> 
struct DEtrait<L1MuDTChambThDigiCollection> {
  typedef L1MuDTChambThDigiCollection coll_type;
  typedef L1MuDTChambThDigi           cand_type;
  static inline int de_type() {return DTtpTh;}
};

template<> 
struct DEtrait<L1MuRegionalCandCollection> {
  typedef L1MuRegionalCandCollection coll_type;
  typedef L1MuRegionalCand           cand_type;
  static inline int de_type() {return 99;}
};
template<> 
struct DEtrait<CSCCorrelatedLCTDigiCollection_> {
  typedef CSCCorrelatedLCTDigiCollection_ coll_type;
  typedef CSCCorrelatedLCTDigi            cand_type;
  static inline int de_type() {return CSCtp;}
};
template<> 
struct DEtrait<L1CSCTrackCollection> {
  typedef L1CSCTrackCollection coll_type;
  typedef L1CSCTrack           cand_type;
  static inline int de_type() {return CSCtf;}
};

template<> 
struct DEtrait<LTCDigiCollection> {
  typedef LTCDigiCollection coll_type;
  typedef LTCDigi           cand_type;
  static inline int de_type() {return LTCi;}
};
template<> 
struct DEtrait<L1MuGMTCandCollection> {
  typedef L1MuGMTCandCollection coll_type;
  typedef L1MuGMTCand           cand_type;
  static inline int de_type() {return GMTcnd;}
};
template<> 
struct DEtrait<L1MuGMTReadoutRecordCollection> {
  typedef L1MuGMTReadoutRecordCollection coll_type;
  typedef L1MuGMTReadoutRecord           cand_type;
  static inline int de_type() {return GMTrdt;}
};

template<> 
struct DEtrait<DecisionWord> {
  typedef DecisionWord coll_type;
  typedef bool                   cand_type;
  static inline int de_type() {return GTdword;}
};

/*
template<> 
struct DEtrait<> {
  typedef coll_type;
  typedef cand_type;
  static inline int de_type() {return 99;}
};
*/

#endif
