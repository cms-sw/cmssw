#ifndef DETRAIT_H
#define DETRAIT_H

/*\class template DEtrait
 *\description data|emulation auxiliary template
               type associator trait struct 
 *\author Nuno Leonardo (CERN)
 *\date 07.03
 */


enum collSet {ECALtp=0, HCALtp, RCTem, RCTrgn, GCTem, GCTjet, GTdword};

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
struct DEtrait<DecisionWord> {
  typedef DecisionWord coll_type;
  typedef bool                   cand_type;
  static inline int de_type() {return GTdword;}
};

#endif
