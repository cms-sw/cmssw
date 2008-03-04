#ifndef DEUTILS_H
#define DEUTILS_H

/*\class template DEutils
 *\description data|emulation auxiliary template
               collection operations struct
 *\author Nuno Leonardo (CERN)
 *\date 07.04
 */

#include "L1Trigger/HardwareValidation/interface/DEtrait.h"

template <typename T> 
struct DEutils {

  typedef typename T::size_type col_sz;
  typedef typename T::const_iterator col_cit;
  typedef typename T::iterator col_it;

  typedef DEtrait<T> de_trait;
  typedef typename de_trait::cand_type cand_type;
  typedef typename de_trait::coll_type coll_type;

  public:
  
  DEutils() {
    if(de_type()>16)
      throw cms::Exception("ERROR") 
	<< "DEutils::DEutils() :: "
	<< "specialization is still missing for collection of type:" 
	<< de_type() << std::endl;
  }
  ~DEutils(){}
  
  inline int de_type() const {return de_trait::de_type();}

  col_it de_find  ( col_it, col_it,  const cand_type&);
  bool   de_equal (const cand_type&, const cand_type&);
  bool   de_nequal(const cand_type&, const cand_type&);
  std::string print(col_cit) const;
  bool is_empty(col_cit) const;
  std::string GetName(int) const;

};


//--//--//--//--//--//--//--//--//--//--//--//

template <typename T>
typename DEutils<T>::col_it DEutils<T>::de_find( col_it first, col_it last, const cand_type& value ) {
  for ( ;first!=last; first++) 
    if ( de_equal(*first,value) ) break;
  return first;
}

//--//--//--//--//--//--//--//--//--//--//--//

template <typename T>
bool DEutils<T>::de_equal(const cand_type& lhs, const cand_type& rhs) {
  return true;
}
template <typename T>
bool DEutils<T>::de_nequal(const cand_type& lhs, const cand_type& rhs) {
  return !de_equal(lhs,rhs);
}

template <>
inline bool DEutils<EcalTrigPrimDigiCollection>::de_equal(const cand_type& lhs, const cand_type& rhs) {
  bool val = true;
  val &= (lhs[lhs.sampleOfInterest()].raw() == rhs[rhs.sampleOfInterest()].raw());
  val &= (lhs.id().rawId()        == rhs.id().rawId());
  //val &= (lhs.id().subDet()  == rhs.id().subDet()); //("Barrel"):("Endcap")) 
  //val &= (lhs.id().zside()   == rhs.id().zside()  );
  //val &= (lhs.id().ietaAbs() == rhs.id().ietaAbs());
  //val &= (lhs.id().iphi()    == rhs.id().iphi()   );
  return val;
  //tbd: add raw data accessor in trigtowerdetid...
}

template <>
inline bool DEutils<HcalTrigPrimDigiCollection>::de_equal(const cand_type& lhs, const cand_type& rhs) {
  bool val = true;
  val &= (lhs.t0().raw()     == rhs.t0().raw());
  val &= (lhs.id().rawId()   == rhs.id().rawId());
  //val &= (lhs.id().subdet()  == rhs.id().subdet());
  //val &= (lhs.id().zside()   == rhs.id().zside()  );
  //val &= (lhs.id().ietaAbs() == rhs.id().ietaAbs());
  //val &= (lhs.id().iphi()    == rhs.id().iphi()   );
  return val;
  //tbd: add raw data accessor in trigtowerdetid...
}

template <>
inline bool DEutils<L1CaloEmCollection>::de_equal(const cand_type& lhs, const cand_type& rhs) {
  bool val = true;
  val &= (lhs.raw()       == rhs.raw()     );
  val &= (lhs.rctCrate()  == rhs.rctCrate());
  val &= (lhs.isolated()  == rhs.isolated());
  return val;
}

template <>
inline bool DEutils<L1CaloRegionCollection>::de_equal(const cand_type& lhs, const cand_type& rhs) {
  bool val = true;

  val &= (lhs.et()        == rhs.et()       );
  val &= (lhs.rctCrate()  == rhs.rctCrate() );	
  val &= (lhs.rctRegionIndex() == rhs.rctRegionIndex());
  val &= (lhs.id().isHf() == rhs.id().isHf());  

  if (!lhs.id().isHf()){
    val &= (lhs.overFlow()  == rhs.overFlow() );
    val &= (lhs.tauVeto()   == rhs.tauVeto()  );
    val &= (lhs.mip()       == rhs.mip()      );
    val &= (lhs.quiet()     == rhs.quiet()    );
    val &= (lhs.rctCard()   == rhs.rctCard()  );
  } else {
    val &= (lhs.fineGrain() == rhs.fineGrain());
  }
  return val;
}

template <>
inline bool DEutils<L1GctEmCandCollection>::de_equal(const cand_type& lhs, const cand_type& rhs) {
  return lhs==rhs;
}
template <>
inline bool DEutils<L1GctJetCandCollection>::de_equal(const cand_type& lhs, const cand_type& rhs) {
  return lhs==rhs;
}

//--//--//--//--//--//--//--//--//--//--//--//

template <typename T> 
bool DEutils<T>::is_empty(col_cit it) const { 
  return true; //default;
}

template<>
inline bool DEutils<EcalTrigPrimDigiCollection>::is_empty(col_cit it) const { 
  return ( it->size()==0 || it->sample(it->sampleOfInterest()).raw()==0);
}

template<>
inline bool DEutils<HcalTrigPrimDigiCollection>::is_empty(col_cit it) const { 
  return (  it->size()==0 ||it->t0().raw()==0 || it->SOI_compressedEt()==0 );
}

template<>
inline bool DEutils<L1CaloEmCollection>::is_empty(col_cit it) const { 
    return  ((it->rank())==0);
    //return  ((it->raw())==0);
}

template<>
inline bool DEutils<L1CaloRegionCollection>::is_empty(col_cit it) const { 
    return  ((it->et())==0);
    //missing accessors to object constructor!
}
template<>
inline bool DEutils<L1GctEmCandCollection>::is_empty(col_cit it) const { 
      return  (it->empty());
}

template<>
inline bool DEutils<L1GctJetCandCollection>::is_empty(col_cit it) const { 
    return  (it->empty());
}

//--//--//--//--//--//--//--//--//--//--//--//

template <typename T> 
std::string DEutils<T>::print(col_cit it) const {
  std::stringstream ss;
  ss << "[DEutils<T>::print()] specialization still missing for collection!";
  //ss << *it; //default
  ss << std::endl;
  return ss.str();
}

template <> 
inline std::string DEutils<EcalTrigPrimDigiCollection>::print(col_cit it) const {
  std::stringstream ss;
  ss << "0x" << std::setw(4) << std::setfill('0') << std::hex 
     << it->sample(it->sampleOfInterest()).raw()
     << std::setfill(' ') << std::dec 
     << " Et:"   << std::setw(3) << it->compressedEt() 
     << " Fg:"   << std::setw(3) << it->fineGrain()
     << " ttf:"  << std::setw(3) << it->ttFlag()
     << " "      << ((it->id().subDet()==EcalBarrel)?("Barrel"):("Endcap")) 
     << " iz:"   << ((it->id().zside()>0)?("+"):("-")) 
     << " ieta:" << std::setw(3) << it->id().ietaAbs()
     << " iphi:" << std::setw(3) << it->id().iphi()
     //<< "\n\traw: " << *it 
     << std::endl;
  return ss.str();
}
template <> 
inline std::string DEutils<HcalTrigPrimDigiCollection>::print(col_cit it) const {
  std::stringstream ss;
  ss << "0x" << std::setw(4) << std::setfill('0') << std::hex 
     << it->t0().raw()
     << std::setfill(' ') << std::dec 
     << " Et:"   << std::setw(3) << it->SOI_compressedEt()
     << " Fg:"   << std::setw(3) << it->SOI_fineGrain()
     << " sdet:" << it->id().subdet()
     << " iz:"   << ((it->id().zside()>0)?("+"):("-")) 
     << " ieta:" << std::setw(3) << it->id().ietaAbs()
     << " iphi:" << std::setw(3) << it->id().iphi()
     << std::endl;
  return ss.str();
}

template <> 
inline std::string DEutils<L1CaloEmCollection>::print(col_cit it) const {
  std::stringstream ss;
  ss << "0x" << std::setw(4) << std::setfill('0') << std::hex 
     << it->raw() 
     << std::setfill(' ') << std::dec << "  "
     << *it
     << std::endl;
  return ss.str();
}
template <> 
inline std::string DEutils<L1CaloRegionCollection>::print(col_cit it) const {
  std::stringstream ss;
  ss << *it << std::endl;
  return ss.str();
}

template <> 
inline std::string DEutils<L1GctEmCandCollection>::print(col_cit it) const {
  std::stringstream ss;
  ss << *it << std::endl;
  return ss.str();
}
template <> 
inline std::string DEutils<L1GctJetCandCollection>::print(col_cit it) const {
  std::stringstream ss;
  ss << *it << std::endl;
  return ss.str();
}

template <> 
inline std::string DEutils<L1MuDTChambPhDigiCollection>::print(col_cit it) const {
  std::stringstream ss;
  ss << ""
     << " bxNum:"  << it->bxNum()  
     << " whNum:"  << it->whNum()  
     << " scNum:"  << it->scNum()  
     << " stNum:"  << it->stNum()  
     << " phi:"    << it->phi()    
     << " phiB:"   << it->phiB()   
     << " code:"   << it->code()   
     << " Ts2Tag:" << it->Ts2Tag() 
     << " BxCnt:"  << it->BxCnt()  
     << std::endl;
  //nb: operator << not implemented in base class L1MuDTChambPhDigi
  return ss.str();
}

template <> 
inline std::string DEutils<L1MuDTChambThDigiCollection>::print(col_cit it) const {
  std::stringstream ss;
  ss << ""
     << " bxNum:"  << it->bxNum()  
     << " whNum:"  << it->whNum()  
     << " scNum:"  << it->scNum()  
     << " stNum:"  << it->stNum()  
     << std::endl;
  //nb: operator << not implemented in base class L1MuDTChambThDigi
  return ss.str();
}


//--//--//--//--//--//--//--//--//--//--//--//

template <typename T> 
std::string DEutils<T>::GetName(int i=0) const {

  const int nlabel = 8;
  if(!(i<nlabel)) 
    return                  "un-defined" ;
  std::string str[nlabel]= {"un-registered"};

  switch(de_type()) {
  case ECALtp:
    str[0] = "ECAL tp";
    str[1] = "EcalTrigPrimDigiCollection";
    str[2] = "EcalTriggerPrimitiveDigi";
  break;
  case HCALtp:
    str[0] = "HCAL tp";
    str[1] = "HcalTrigPrimDigiCollection";
    str[2] = "HcalTriggerPrimitiveDigi";
  break;
  case RCTem:
    str[0] = "RCT em";
    str[1] = "L1CaloEmCollection";
    str[2] = "L1CaloEmCand";
  break;
  case RCTrgn:
    str[0] = "RCT region";
    str[1] = "L1CaloRegionCollection";
    str[2] = "L1CaloRegion";
    break;
  case GCTem:
    str[0] = "GCT em";
    str[1] = "L1GctEmCandCollection";
    str[2] = "L1GctEmCand";
   break;
  case GCTjet:
    str[0] = "GCT jet";
    str[1] = "L1GctJetCandCollection";
    str[2] = "L1GctJetCand";
   break;
  case DTtpPh:
    str[0] = "DT tp phi";
    str[1] = "L1MuDTChambPhDigiCollection";
    str[2] = "L1MuDTChambPhDigi";
   break;
  case DTtpTh:
    str[0] = "DT tp theta";
    str[1] = "L1MuDTChambThDigiCollection";
    str[2] = "L1MuDTChambThDigi";
   break;
  case CSCtp:
    str[0] = "CSC tp";
    str[1] = "CSCCorrelatedLCTDigiCollection";
    str[2] = "CSCCorrelatedLCTDigi";
   break;
  case CSCtf:
    str[0] = "CSC track";
    str[1] = "L1CSCTrackCollection";
    str[2] = "L1CSCTrack";
   break;
  case RPCb:
    str[0] = "RPC barrel";
    str[1] = "L1MuRegionalCandCollection";
    str[2] = "L1MuRegionalCand";
   break;
  case RPCf:
    str[0] = "RPC tf forward";
    str[1] = "L1MuRegionalCandCollection";
    str[2] = "L1MuRegionalCand";
   break;
  case LTCi:
    str[0] = "LTC";
    str[1] = "LTCDigiCollection";
    str[2] = "LTCDigi";
   break;
  case GMTcnd:
    str[0] = "GMT cand";
    str[1] = "L1MuGMTCandCollection";
    str[2] = "L1MuGMTCand";
   break;
  case GMTrdt:
    str[0] = "GMT record";
    str[1] = "L1MuGMTReadoutRecordCollection";
    str[2] = "L1MuGMTReadoutRecord";
   break;
  case GTdword:
    str[0] = "";
    str[1] = "";
    str[2] = "";
   break;
   //default:
  }
  return str[i];
}

//--//--//--//--//--//--//--//--//--//--//--//
//--//--//--//--//--//--//--//--//--//--//--//

template <typename T>
struct de_rank : public DEutils<T> , public std::binary_function<typename DEutils<T>::cand_type, typename DEutils<T>::cand_type, bool> {
  typedef DEtrait<T> de_trait;
  typedef typename de_trait::cand_type cand_type;
  bool operator()(const cand_type& x, const cand_type& y) const {
    return true; //default
  }
};

template <> inline bool de_rank<EcalTrigPrimDigiCollection>::operator()(const cand_type& x, const cand_type& y) const { return x.compressedEt() > y.compressedEt(); }
template <> inline bool de_rank<HcalTrigPrimDigiCollection>::operator()(const cand_type& x, const cand_type& y) const { return x.SOI_compressedEt() > y.SOI_compressedEt(); }

template <> 
inline bool de_rank<L1CaloEmCollection>::operator() 
     (const cand_type& x, const cand_type& y) const {
  if       (x.rank()      != y.rank())     {
    return (x.rank()      >  y.rank())     ;
  } else if(x.isolated()  != y.isolated()) {
    return (x.isolated())?1:0;
  } else if(x.rctRegion() != y.rctRegion()){
    return (x.rctRegion() <  y.rctRegion());
  } else if(x.rctCrate()  != y.rctCrate()) {
    return (x.rctCrate()  <  y.rctCrate()) ;
  } else if(x.rctCard()   != y.rctCard())  {
    return (x.rctCard()   <  y.rctCard())  ;
  } else {
    return x.raw() < y.raw();
  }
}

template <> inline bool de_rank<L1CaloRegionCollection>::operator()(const cand_type& x, const cand_type& y) const { return x.et() > y.et(); }

template <> inline bool de_rank<L1GctEmCandCollection>::operator()(const cand_type& x, const cand_type& y)const { if(x.rank()!=y.rank()){return x.rank() > y.rank();} else{if(x.etaIndex()!=y.etaIndex()){return y.etaIndex() > x.etaIndex();}else{ return x.phiIndex() > y.phiIndex();}}}

template <> inline bool de_rank<L1GctJetCandCollection>::operator()(const cand_type& x, const cand_type& y)const { if(x.rank()!=y.rank()){return x.rank() > y.rank();} else{if(x.etaIndex()!=y.etaIndex()){return y.etaIndex() > x.etaIndex();}else{ return x.phiIndex() > y.phiIndex();}}}


#endif
