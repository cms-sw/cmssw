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
    if(de_type()>51)
      edm::LogError("L1ComparatorDeutilsCollType") //throw cms::Exception("ERROR") 
	<< "DEutils::DEutils() :: "
	<< "specialization is still missing for collection of type:" 
	<< de_type() << std::endl;
  }
  ~DEutils(){}
  
  inline int de_type() const {return de_trait::de_type();}
  bool   de_equal     (const cand_type&, const cand_type&);
  bool   de_equal_loc (const cand_type&, const cand_type&);
  bool   de_nequal    (const cand_type&, const cand_type&);
  bool   de_nequal_loc(const cand_type&, const cand_type&);
  col_it de_find      ( col_it, col_it,  const cand_type&);
  //col_it de_find_loc  ( col_it, col_it,  const cand_type&);

  std::string print(col_cit) const;
  bool is_empty(col_cit) const;
  std::string GetName(int i = 0) const;

  L1DataEmulDigi DEDigi(col_cit itd, col_cit itm, int ctype);
  
};


/// --- form de-digi ---

template <typename T> 
L1DataEmulDigi DEutils<T>::DEDigi(col_cit itd,  col_cit itm, int aflag) {
  ///return empty digi by default
  return L1DataEmulDigi();
}

template<> inline L1DataEmulDigi 
DEutils<EcalTrigPrimDigiCollection>::DEDigi(col_cit itd,  col_cit itm, int aflag) {
  int cid = de_type();
  int errt = aflag;
  //fill data if flagged, otherwise emulator
  double x1 = (aflag!=4) ? itd->id().iphi() : itm->id().iphi();
  double x2 = (aflag!=4) ? itd->id().ieta() : itm->id().ieta();
  L1DataEmulDigi digi(dedefs::ETP,cid, x1,x2,0, errt);
  unsigned int dwS = (aflag==4)?0:itd->sample(itd->sampleOfInterest()).raw();
  unsigned int ewS = (aflag==3)?0:itm->sample(itm->sampleOfInterest()).raw();
  //dw1 &= 0x01ff; ew1 &= 0x01ff; //9-bit: fg(8),energy(7:0)
  unsigned int mask = 0x0eff; //fg bit temporary(!) mask
  dwS &= mask;   ewS &= mask; 
  unsigned int dwI = (aflag==4)?0:itd->id().rawId();
  unsigned int ewI = (aflag==3)?0:itm->id().rawId();
  //dw2 &= 0xfe00ffff; ew2 &= 0xfe00ffff; //32-bit, reset unused (24:16)
  //merge id and sample words
  unsigned int dw = (dwI & 0xfe00ffff ) | ( (dwS & 0x000001ff)<<16 ); 
  unsigned int ew = (ewI & 0xfe00ffff ) | ( (ewS & 0x000001ff)<<16 );
  digi.setData(dw,ew);
  int de = (aflag==4)?0:itd->compressedEt() ;
  int ee = (aflag==3)?0:itm->compressedEt() ;
  digi.setRank((float)de,(float)ee);
  L1MonitorDigi dedata(dedefs::ETP,cid, itd->id().iphi(),itd->id().ieta(),0, 
		       itd->compressedEt(),itd->id().rawId());
  L1MonitorDigi deemul(dedefs::ETP,cid, itm->id().iphi(),itm->id().ieta(),0, 
		       itm->compressedEt(),itm->id().rawId());
  digi.setDEpair(dedata,deemul);
  return digi;
}

template<> inline L1DataEmulDigi 
DEutils<HcalTrigPrimDigiCollection>::DEDigi(col_cit itd,  col_cit itm, int aflag) {
  int cid = de_type();
  int errt = aflag;
  double x1 = (aflag!=4) ? itd->id().iphi() : itm->id().iphi();
  double x2 = (aflag!=4) ? itd->id().ieta() : itm->id().ieta();
  L1DataEmulDigi digi(dedefs::HTP,cid, x1,x2,0, errt);
  unsigned int dw = (aflag==4)?0:itd->t0().raw();
  unsigned int ew = (aflag==3)?0:itm->t0().raw();
  //16-bit; bits 10:9 not set(?); 
  // bits 15:11 not accessible in emulator (slb/channel ids)
  unsigned int mask = 0x01ff;
  dw &= mask; ew &= mask; 
  digi.setData(dw,ew); 
  int de = (aflag==4)?0:itd->SOI_compressedEt();
  int ee = (aflag==3)?0:itm->SOI_compressedEt();
  digi.setRank((float)de,(float)ee);
  return digi;
}

template<> inline L1DataEmulDigi 
DEutils<L1CaloEmCollection>::DEDigi(col_cit itd,  col_cit itm, int aflag) {
  int cid = de_type();
  int errt = aflag;
  double x1, x2, x3(0.);
  // global index ieta (0-21), iphi (0-17), card (0-6)
  x1 = (aflag!=4) ? itd->regionId().iphi() : itm->regionId().iphi();
  x2 = (aflag!=4) ? itd->regionId().ieta() : itm->regionId().ieta();
  x3 = (aflag!=4) ? itd->regionId().rctCard() : itm->regionId().rctCard();
  //alternative coordinates: rctCrate(), rctCard(), index()
  L1DataEmulDigi digi(dedefs::RCT,cid, x1,x2,x3, errt);
  unsigned int dw = itd->raw(); 
  unsigned int ew = itm->raw();
  dw &= 0x3ff;
  dw += (((itd->rctCrate())&0x1f)<<10);
  dw += (((itd->isolated()?1:0)&0x1)<<15);
  dw += (((itd->index())&0x3)<<16);
  ew &= 0x3ff;
  ew += (((itm->rctCrate())&0x1f)<<10);
  ew += (((itm->isolated()?1:0)&0x1)<<15);
  ew += (((itm->index())&0x3)<<16);
  dw = (aflag==4)?0:dw;
  ew = (aflag==3)?0:ew;
  /// bits: index(17:16) iso(15) crate(14:10)  +  card(9:7) region(6) rank (5:0)
  /// (rank & 0x3f) + ((region & 0x1)<<6) + ((card & 0x7)<<7)
  ///  + ((card & 0x1f)<<10) + ((0x1)<<15) + ((0x3)<<16)
  digi.setData(dw,ew); 
  int de = (aflag==4)?0:itd->rank();
  int ee = (aflag==3)?0:itm->rank();
  digi.setRank((float)de,(float)ee);
  return digi;
}

template<> inline L1DataEmulDigi 
DEutils<L1CaloRegionCollection>::DEDigi(col_cit itd,  col_cit itm, int aflag) {
  int cid = de_type();
  int errt = aflag;
  double x1, x2, x3(0.);
  x1 = (aflag!=4) ? itd->id().iphi() : itm->id().iphi();
  x2 = (aflag!=4) ? itd->id().ieta() : itm->id().ieta();
  x3 = (aflag!=4) ? itd->id().rctCard() : itm->id().rctCard();
  L1DataEmulDigi digi(dedefs::RCT,cid, x1,x2,x3, errt);
  unsigned int dw = itd->raw(); 
  unsigned int ew = itm->raw();
  unsigned int mask = 0x3fff;
  //mask (temporary) mip(12), quiet (13)
  mask = 0x0fff;
  dw &= mask;
  dw += (((itd->id().ieta())&0x1f)<<14);
  dw += (((itd->id().iphi())&0x1f)<<19);
  ew &= mask;
  ew += (((itm->id().ieta())&0x1f)<<14);
  ew += (((itm->id().iphi())&0x1f)<<19);
  dw = (aflag==4)?0:dw;
  ew = (aflag==3)?0:ew;
  /// bits: iphi(23:19), ieta(18:14) + quiet(13), mip(12), fg(11), ovf(10), et (9:0)
  digi.setData(dw,ew); 
  int de = (aflag==4)?0:itd->et();
  int ee = (aflag==3)?0:itm->et();
  digi.setRank((float)de,(float)ee);
  return digi;
}

template<> inline L1DataEmulDigi 
DEutils<L1GctEmCandCollection>::DEDigi(col_cit itd,  col_cit itm, int aflag) {
  int cid = de_type();
  int errt = aflag;
  //phi: 0..17; eta: 0..21
  // bring it to global coordinates
  double x1 = (aflag!=4) ? itd->regionId().iphi() : itm->regionId().iphi();
  double x2 = (aflag!=4) ? itd->regionId().ieta() : itm->regionId().ieta();
  L1DataEmulDigi digi(dedefs::GCT,cid, x1,x2,0., errt);
  unsigned int dw = (aflag==4)?0:itd->raw();
  unsigned int ew = (aflag==3)?0:itm->raw();
  dw &= 0x7fff; ew &= 0x7fff; //15-bit
  digi.setData(dw,ew); 
  int de = (aflag==4)?0:itd->rank();
  int ee = (aflag==3)?0:itm->rank();
  digi.setRank((float)de,(float)ee);
  return digi;
}

template<> inline L1DataEmulDigi 
DEutils<L1GctJetCandCollection>::DEDigi(col_cit itd,  col_cit itm, int aflag) {
  int cid = de_type();
  int errt = aflag;
  //phi: 0..17; eta: -6..-0,+0..+6; eta sign:1(z-),0(z+)
  // bring it to global coordinates 0..21 below
  double x1 = (aflag!=4) ? itd->phiIndex() : itm->phiIndex();
  unsigned deta(0), eeta(0);
  if (!itd->isForward()) deta=(itd->etaSign()==1?10-(itd->etaIndex()&0x7):(itd->etaIndex()&0x7)+11);
  else                   deta=(itd->etaSign()==1? 3-(itd->etaIndex()&0x7):(itd->etaIndex()&0x7)+18 );
  if (!itm->isForward()) eeta=(itm->etaSign()==1?10-(itm->etaIndex()&0x7):(itm->etaIndex()&0x7)+11);
  else                   eeta=(itm->etaSign()==1? 3-(itm->etaIndex()&0x7):(itm->etaIndex()&0x7)+18 );
  double x2 = (aflag!=4) ? deta : eeta;
  L1DataEmulDigi digi(dedefs::GCT,cid, x1,x2,0., errt);
  unsigned int dw = (aflag==4)?0:itd->raw();
  unsigned int ew = (aflag==3)?0:itm->raw();
  dw &= 0x7fff; ew &= 0x7fff; //15-bit
  digi.setData(dw,ew); 
  int de = (aflag==4)?0:itd->rank();
  int ee = (aflag==3)?0:itm->rank();
  digi.setRank((float)de,(float)ee);
  return digi;
}

template<> inline L1DataEmulDigi 
DEutils<L1GctEtHadCollection>::DEDigi(col_cit itd,  col_cit itm, int aflag) {
  int cid = de_type();
  int errt = aflag;
  double x1 = 0;  double x2 = 0; //no 'location' associated with candidates...
  L1DataEmulDigi digi(dedefs::GCT,cid, x1,x2,0., errt);
  unsigned int dw = (aflag==4)?0:itd->raw();
  unsigned int ew = (aflag==3)?0:itm->raw();
  dw &= 0x1fff; ew &= 0x1fff; //13-bit
  digi.setData(dw,ew); 
  int de = (aflag==4)?0:itd->et();
  int ee = (aflag==3)?0:itm->et();
  digi.setRank((float)de,(float)ee);
  return digi;
}
template<> inline L1DataEmulDigi 
DEutils<L1GctEtMissCollection>::DEDigi(col_cit itd,  col_cit itm, int aflag) {
  int cid = de_type();
  int errt = aflag;
  double x1 = (aflag!=4) ? itd->phi() : itm->phi();
  double x2 = 0; //no 'eta' associated with candidates... 
  L1DataEmulDigi digi(dedefs::GCT,cid, x1,x2,0., errt);
  unsigned int dw = (aflag==4)?0:itd->raw();
  unsigned int ew = (aflag==3)?0:itm->raw();
  dw &= 0x8f1fff; ew &= 0x8f1fff; //22-bit (bits 13,14,15 not set)
  digi.setData(dw,ew); 
  int de = (aflag==4)?0:itd->et();
  int ee = (aflag==3)?0:itm->et();
  digi.setRank((float)de,(float)ee);
  return digi;
}

template<> inline L1DataEmulDigi 
DEutils<L1GctEtTotalCollection>::DEDigi(col_cit itd,  col_cit itm, int aflag) {
  int cid = de_type();
  int errt = aflag;
  double x1 = 0;  double x2 = 0; //no 'location' associated with candidates...
  L1DataEmulDigi digi(dedefs::GCT,cid, x1,x2,0., errt);
  unsigned int dw = (aflag==4)?0:itd->raw();
  unsigned int ew = (aflag==3)?0:itm->raw();
  dw &= 0x1fff; ew &= 0x1fff; //13-bit
  digi.setData(dw,ew); 
  int de = (aflag==4)?0:itd->et();
  int ee = (aflag==3)?0:itm->et();
  digi.setRank((float)de,(float)ee);
  return digi;
}
template<> inline L1DataEmulDigi 
DEutils<L1GctHFBitCountsCollection>::DEDigi(col_cit itd,  col_cit itm, int aflag) {
  int cid = de_type();
  int errt = aflag;
  double x1 = 0;  double x2 = 0; //no 'location' associated with candidates...
  L1DataEmulDigi digi(dedefs::GCT,cid, x1,x2,0., errt);
  unsigned int dw = (aflag==4)?0:itd->raw();
  unsigned int ew = (aflag==3)?0:itm->raw();
  digi.setData(dw,ew); 
  int de = 0;  int ee = 0; //no 'rank' associated with candidates...
  digi.setRank((float)de,(float)ee);
  return digi;
}
template<> inline L1DataEmulDigi 
DEutils<L1GctHFRingEtSumsCollection>::DEDigi(col_cit itd,  col_cit itm, int aflag) {
  int cid = de_type();
  int errt = aflag;
  double x1 = 0;  double x2 = 0; //no 'location' associated with candidates...
  L1DataEmulDigi digi(dedefs::GCT,cid, x1,x2,0., errt);
  unsigned int dw = (aflag==4)?0:itd->raw();
  unsigned int ew = (aflag==3)?0:itm->raw();
  digi.setData(dw,ew); 
  int de = 0;  int ee = 0; //no 'rank' associated with candidates...
  digi.setRank((float)de,(float)ee);
  return digi;
}
template<> inline L1DataEmulDigi 
DEutils<L1GctHtMissCollection>::DEDigi(col_cit itd,  col_cit itm, int aflag) {
  int cid = de_type();
  int errt = aflag;
  double x1 = (aflag!=4) ? itd->phi() : itm->phi();
  double x2 = 0; //no 'eta' associated with candidates... 
  L1DataEmulDigi digi(dedefs::GCT,cid, x1,x2,0., errt);
  unsigned int dw = (aflag==4)?0:itd->raw();
  unsigned int ew = (aflag==3)?0:itm->raw();
  dw &= 0x8f1fff; ew &= 0x8f1fff; //22-bit (bits 13,14,15 not set)
  digi.setData(dw,ew); 
  int de = (aflag==4)?0:itd->et();
  int ee = (aflag==3)?0:itm->et();
  digi.setRank((float)de,(float)ee);
  return digi;
}

template<> inline L1DataEmulDigi 
DEutils<L1GctJetCountsCollection>::DEDigi(col_cit itd,  col_cit itm, int aflag) {
  int cid = de_type();
  int errt = aflag;
  double x1 = 0;  double x2 = 0; //no 'location' associated with candidates...
  L1DataEmulDigi digi(dedefs::GCT,cid, x1,x2,0., errt);
  unsigned int dw = (aflag==4)?0:itd->raw0();//raw0, raw1...
  unsigned int ew = (aflag==3)?0:itm->raw0();//raw0, raw1...
  digi.setData(dw,ew); 
  int de = 0;  int ee = 0; //no 'rank' associated with candidates...
  digi.setRank((float)de,(float)ee);
  return digi;
}

template<> inline L1DataEmulDigi 
DEutils<L1MuRegionalCandCollection>::DEDigi(col_cit itd,  col_cit itm, int aflag) {
  int sid;
  switch(itd->type_idx()) { // 0 DT, 1 bRPC, 2 CSC, 3 fRPC
  case 0:  sid=dedefs::DTF; break;
  case 1:  sid=dedefs::RPC; break;
  case 2:  sid=dedefs::CTF; break;
  case 3:  sid=dedefs::RPC; break;
  default: sid=-1;
  }
  int cid = de_type();
  int errt = aflag;
  //double x1 = (aflag!=4) ? itd->phiValue() : itm->phiValue();
  //double x2 = (aflag!=4) ? itd->etaValue() : itm->etaValue();
  double x1 = (aflag!=4) ? itd->phi_packed() : itm->phi_packed();
  double x2 = (aflag!=4) ? itd->eta_packed() : itm->eta_packed();
  L1DataEmulDigi digi(sid,cid, x1,x2,0, errt);
  unsigned int dw = (aflag==4)?0 : itd->getDataWord();
  unsigned int ew = (aflag==3)?0 : itm->getDataWord();
  unsigned int mask = 0xffffffff; //32-bit
  //RPC: mask bits 25-29 (including synch bits)
  // emulator doesn't set these bits (permanent masking)
  if(sid==dedefs::RPC)
    mask &= 0xc1ffffff;
  dw &= mask; ew &= mask;
  digi.setData(dw,ew);
  int de = (aflag==4)?0:itd->pt_packed();//ptValue();
  int ee = (aflag==3)?0:itm->pt_packed();//ptValue();
  digi.setRank((float)de,(float)ee);
  //note: phi,eta,pt 'values' not always set for all muon tf systems
  //(under discussion) need universal mechanism for setting up physical units
  if(0) //check print
    std::cout << "L1DataEmulDigi DEutils<L1MuRegionalCandCollection>] dedigi info"
      //<< " phivalue:" << itd->phiValue()   << "," << itm->phiValue()
      //<< " etavalue:" << itd->etaValue()   << "," << itm->etaValue()
	      << " phipackd:" << itd->phi_packed() << "," << itm->phi_packed()
	      << " etapackd:" << itd->eta_packed() << "," << itm->eta_packed()
	      << std::endl;
  return digi;
}

template<> inline L1DataEmulDigi 
DEutils<L1MuGMTCandCollection>::DEDigi(col_cit itd,  col_cit itm, int aflag) {
  int cid = de_type();
  int errt = aflag;
  //double x1 = (aflag!=4) ? itd->phiValue() : itm->phiValue();
  //double x2 = (aflag!=4) ? itd->etaValue() : itm->etaValue();
  double x1 = (aflag!=4) ? itd->phiIndex() : itm->phiIndex();
  double x2 = (aflag!=4) ? itd->etaIndex() : itm->etaIndex();
  L1DataEmulDigi digi(dedefs::GMT,cid, x1,x2,0, errt);
  unsigned int dw = (aflag==4)?0 : itd->getDataWord();
  unsigned int ew = (aflag==3)?0 : itm->getDataWord();
  unsigned int mask = 0x3ffffff; //26-bit
  //mask bits 22 (isolation), 23 (mip) [not permanent!]
  mask &= (~(0x0c00000)); 
  dw &= mask; ew &= mask;
  digi.setData(dw,ew);
  int de = (aflag==4)?0:itd->ptIndex();//ptValue();
  int ee = (aflag==3)?0:itm->ptIndex();//ptValue();
  digi.setRank((float)de,(float)ee);
  if(0) //check print
  std::cout << "l1dataemuldigi l1mugmtcandcoll type:" << cid 
    //<< " eta:" << itd->etaValue() << ", " << itm->etaValue()
    //<< " phi:" << itd->phiValue() << ", " << itm->phiValue()
	    << std::hex << " word d:" << dw << "e:" << ew << std::dec 
	    << std::endl;
  return digi;
}

template<> 
inline L1DataEmulDigi DEutils<L1MuDTChambPhDigiCollection>::DEDigi(col_cit itd,  col_cit itm, int aflag) {
  int cid = de_type();
  int errt = aflag;
  double x1 = (aflag!=4) ? itd->scNum() : itm->scNum();
  double x2 = (aflag!=4) ? itd->whNum() : itm->whNum();
  double x3 = (aflag!=4) ? itd->stNum() : itm->stNum();
  L1DataEmulDigi digi(dedefs::DTP,cid, x1,x2,x3, errt);
  //other coordinate methods phi(), phiB()
  //note: no data word defined for candidate
  int dr = (aflag==4)?0:itd->code();
  int er = (aflag==3)?0:itm->code();
  digi.setRank((float)dr,(float)er);
  return digi;
}

template<> inline L1DataEmulDigi 
DEutils<L1MuDTChambThDigiCollection>::DEDigi(col_cit itd,  col_cit itm, int aflag) {
  int cid = de_type();
  int errt = aflag;
  double x1 = (aflag!=4) ? itd->scNum() : itm->scNum();
  double x2 = (aflag!=4) ? itd->whNum() : itm->whNum();
  double x3 = (aflag!=4) ? itd->stNum() : itm->stNum();
  L1DataEmulDigi digi(dedefs::DTP,cid, x1,x2,x3, errt);
  //note: no data word defined for candidate
  int dr(0), er(0);
  for(int i=0; i<7;i++){
    if(itd->code(i)>=dr) dr=itd->quality(i);
    if(itm->code(i)>=er) er=itm->quality(i);
  }
  //alternatives: code() = quality() + positions()
  dr = (aflag==4)?0:dr;
  er = (aflag==3)?0:er;
  digi.setRank((float)dr,(float)er);
  return digi;
}


template<> inline L1DataEmulDigi 
DEutils<CSCCorrelatedLCTDigiCollection_>::DEDigi(col_cit itd,  col_cit itm, int aflag) {
  int cid = de_type();
  int errt = aflag;
  double x1 = (aflag!=4) ? itd->getStrip() : itm->getStrip();
  double x2 = (aflag!=4) ? itd->getKeyWG() : itm->getKeyWG();
  double x3 = (aflag!=4) ? itd->getTrknmb(): itm->getTrknmb();
  //multiple subsystem ctp,ctf
  L1DataEmulDigi digi(-1,cid, x1,x2,x3, errt);
  int dq = (aflag==4)?0:itd->getQuality();
  int eq = (aflag==3)?0:itm->getQuality();
  digi.setRank((float)dq,(float)eq);
 // Pack LCT digi members into 32-bit data words.
  static const int kValidBitWidth     = 1; // Reverse the order of the 1st
  static const int kQualityBitWidth   = 4; // frame to keep the valid bit
  static const int kPatternBitWidth   = 4; // first and quality second, as
  static const int kWireGroupBitWidth = 7; // is done in ALCT and CLCT.
  static const int kHalfstripBitWidth = 8;
  static const int kBendBitWidth      = 1;
  static const int kBxBitWidth        = 1;
  // Use sync_err and bx0_local bits to store MPC link.
  static const int kMPCLinkBitWidth   = 2;
  static const int kCSCIdBitWidth     = 4;
  // While packing, check that the right number of bits is retained.
  unsigned shift = 0, dw = 0, ew = 0;
  dw  =  itd->isValid()    & ((1<<kValidBitWidth)-1);
  dw += (itd->getQuality() & ((1<<kQualityBitWidth)-1))   <<
    (shift += kValidBitWidth);
  dw += (itd->getPattern() & ((1<<kPatternBitWidth)-1))   <<
    (shift += kQualityBitWidth);
  dw += (itd->getKeyWG()   & ((1<<kWireGroupBitWidth)-1)) <<
    (shift += kPatternBitWidth);
  dw += (itd->getStrip()   & ((1<<kHalfstripBitWidth)-1)) <<
    (shift += kWireGroupBitWidth);
  dw += (itd->getBend()    & ((1<<kBendBitWidth)-1))      <<
    (shift += kHalfstripBitWidth);
  dw += (itd->getBX()      & ((1<<kBxBitWidth)-1))        <<
    (shift += kBendBitWidth);
  dw += (itd->getMPCLink() & ((1<<kMPCLinkBitWidth)-1))   <<
    (shift += kBxBitWidth);
  dw += (itd->getCSCID()   & ((1<<kCSCIdBitWidth)-1))     <<
    (shift += kMPCLinkBitWidth);
  shift = 0;
  ew  =  itm->isValid()    & ((1<<kValidBitWidth)-1);
  ew += (itm->getQuality() & ((1<<kQualityBitWidth)-1))   <<
    (shift += kValidBitWidth);
  ew += (itm->getPattern() & ((1<<kPatternBitWidth)-1))   <<
    (shift += kQualityBitWidth);
  ew += (itm->getKeyWG()   & ((1<<kWireGroupBitWidth)-1)) <<
    (shift += kPatternBitWidth);
  ew += (itm->getStrip()   & ((1<<kHalfstripBitWidth)-1)) <<
    (shift += kWireGroupBitWidth);
  ew += (itm->getBend()    & ((1<<kBendBitWidth)-1))      <<
    (shift += kHalfstripBitWidth);
  ew += (itm->getBX()      & ((1<<kBxBitWidth)-1))        <<
    (shift += kBendBitWidth);
  ew += (itm->getMPCLink() & ((1<<kMPCLinkBitWidth)-1))   <<
    (shift += kBxBitWidth);
  ew += (itm->getCSCID()   & ((1<<kCSCIdBitWidth)-1))     <<
    (shift += kMPCLinkBitWidth);
  digi.setData(dw, ew);
  return digi;
}

template<> inline L1DataEmulDigi 
DEutils<CSCALCTDigiCollection_>::DEDigi(col_cit itd,  col_cit itm, int aflag) {
  int cid = de_type();
  int errt = aflag;
  double x2 = (aflag!=4) ? itd->getKeyWG () : itm->getKeyWG ();
  double x3 = (aflag!=4) ? itd->getTrknmb() : itm->getTrknmb();
  L1DataEmulDigi digi(dedefs::CTP,cid, 0,x2,x3, errt);
  int dq = (aflag==4)?0:itd->getQuality();
  int eq = (aflag==3)?0:itm->getQuality();
  digi.setRank((float)dq,(float)eq);
  // Pack anode digi members into 17-bit data words.
  static const int kValidBitWidth     = 1;
  static const int kQualityBitWidth   = 2;
  static const int kAccelBitWidth     = 1;
  static const int kPatternBBitWidth  = 1;
  static const int kWireGroupBitWidth = 7;
  static const int kBxBitWidth        = 5;
  // While packing, check that the right number of bits is retained.
  unsigned shift = 0, dw = 0, ew = 0;
  dw  =  itd->isValid()        & ((1<<kValidBitWidth)-1);
  dw += (itd->getQuality()     & ((1<<kQualityBitWidth)-1))   <<
    (shift += kValidBitWidth);
  dw += (itd->getAccelerator() & ((1<<kAccelBitWidth)-1))     <<
    (shift += kQualityBitWidth);
  dw += (itd->getCollisionB()  & ((1<<kPatternBBitWidth)-1))  <<
    (shift += kAccelBitWidth);
  dw += (itd->getKeyWG()       & ((1<<kWireGroupBitWidth)-1)) <<
    (shift += kPatternBBitWidth);
  dw += (itd->getBX()          & ((1<<kBxBitWidth)-1))        <<
    (shift += kWireGroupBitWidth);
  shift = 0;
  ew  =  itm->isValid()        & ((1<<kValidBitWidth)-1);
  ew += (itm->getQuality()     & ((1<<kQualityBitWidth)-1))   <<
    (shift += kValidBitWidth);
  ew += (itm->getAccelerator() & ((1<<kAccelBitWidth)-1))     <<
    (shift += kQualityBitWidth);
  ew += (itm->getCollisionB()  & ((1<<kPatternBBitWidth)-1))  <<
    (shift += kAccelBitWidth);
  ew += (itm->getKeyWG()       & ((1<<kWireGroupBitWidth)-1)) <<
    (shift += kPatternBBitWidth);
  ew += (itm->getBX()          & ((1<<kBxBitWidth)-1))        <<
    (shift += kWireGroupBitWidth);
  digi.setData(dw, ew);
  return digi;
}
template<> inline L1DataEmulDigi 
DEutils<CSCCLCTDigiCollection_>::DEDigi(col_cit itd,  col_cit itm, int aflag) {
  int cid = de_type();
  int errt = aflag;
  double x1 = (aflag!=4) ? itd->getKeyStrip() : itm->getKeyStrip();
  double x3 = (aflag!=4) ? itd->getTrknmb() : itm->getTrknmb();
  L1DataEmulDigi digi(dedefs::CTP,cid, x1,0,x3, errt);
  int dq = (aflag==4)?0:itd->getQuality();
  int eq = (aflag==3)?0:itm->getQuality();
  digi.setRank((float)dq,(float)eq);
  // Pack cathode digi members into 19-bit data words.
  static const int kValidBitWidth     = 1;
  static const int kQualityBitWidth   = 3;
  static const int kPatternBitWidth   = 4;
  static const int kBendBitWidth      = 1;
  static const int kHalfstripBitWidth = 5;
  static const int kCFEBBitWidth      = 3;
  static const int kBxBitWidth        = 2;
  // While packing, check that the right number of bits is retained.
  unsigned shift = 0, dw = 0, ew = 0;
  dw  =  itd->isValid()    & ((1<<kValidBitWidth)-1);
  dw += (itd->getQuality() & ((1<<kQualityBitWidth)-1))   <<
    (shift += kValidBitWidth);
  dw += (itd->getPattern() & ((1<<kPatternBitWidth)-1))   <<
    (shift += kQualityBitWidth);
  dw += (itd->getBend()    & ((1<<kBendBitWidth)-1))      <<
    (shift += kPatternBitWidth);
  dw += (itd->getStrip()   & ((1<<kHalfstripBitWidth)-1)) <<
    (shift += kBendBitWidth);
  dw += (itd->getCFEB()    & ((1<<kCFEBBitWidth)-1))      <<
    (shift += kHalfstripBitWidth);
  dw += (itd->getBX()      & ((1<<kBxBitWidth)-1))        <<
    (shift += kCFEBBitWidth);
  shift = 0;
  ew  =  itm->isValid()    & ((1<<kValidBitWidth)-1);
  ew += (itm->getQuality() & ((1<<kQualityBitWidth)-1))   <<
    (shift += kValidBitWidth);
  ew += (itm->getPattern() & ((1<<kPatternBitWidth)-1))   <<
    (shift += kQualityBitWidth);
  ew += (itm->getBend()    & ((1<<kBendBitWidth)-1))      <<
    (shift += kPatternBitWidth);
  ew += (itm->getStrip()   & ((1<<kHalfstripBitWidth)-1)) <<
    (shift += kBendBitWidth);
  ew += (itm->getCFEB()    & ((1<<kCFEBBitWidth)-1))      <<
    (shift += kHalfstripBitWidth);
  ew += (itm->getBX()      & ((1<<kBxBitWidth)-1))        <<
    (shift += kCFEBBitWidth);
  digi.setData(dw, ew);
  return digi;
}

template<> inline L1DataEmulDigi 
DEutils<L1CSCSPStatusDigiCollection_>::DEDigi(col_cit itd,  col_cit itm, int aflag) {
  int cid = de_type();
  int errt = aflag;
  double x1; //sector/slot
  x1 = (aflag!=4) ? itd->slot() : itm->slot();
  //sector-slot map to be in principle to be provided from event setup
  //int de_cscstatus_slot2sector[22] = 
  // {0,0,0,0,0, 0,1,2,3,4, 5,6,0,0,0, 0,7,8,9,10,  11,12};
  //x1 = (aflag!=4) ? slot2sector[itd->slot()] : slot2sector[itm->slot()];
  L1DataEmulDigi digi(dedefs::CTF,cid, x1,0,0, errt);
  //note: no data word and rank defined for candidate
  return digi;
}

/// --- find candidate ---

template <typename T> typename 
DEutils<T>::col_it DEutils<T>::de_find( col_it first, col_it last, const cand_type& value ) {
  for ( ;first!=last; first++) 
    if ( de_equal(*first,value) ) break;
  return first;
}

/*
template <typename T> typename 
DEutils<T>::col_it DEutils<T>::de_find_loc( col_it first, col_it last, const cand_type& value ) {
  for ( ;first!=last; first++) 
    if ( de_equal_loc(*first,value) ) break;
  return first;
}
*/

/// --- candidate match definition ---

template <typename T>
bool DEutils<T>::de_equal(const cand_type& lhs, const cand_type& rhs) {
  //declare candidate matching by default
  return true;
}
template <typename T>
bool DEutils<T>::de_nequal(const cand_type& lhs, const cand_type& rhs) {
  return !de_equal(lhs,rhs);
}

template <> inline bool 
DEutils<EcalTrigPrimDigiCollection>::de_equal(const cand_type& lhs, const cand_type& rhs) {
  bool val = true;
  unsigned int mask = 0x0fff; //keep only ttf[11:9], fg [8], Et [7:0]
  mask &= 0x0eff; //fg bit temporary(!) mask
  val &= ((lhs[lhs.sampleOfInterest()].raw()&mask) == (rhs[rhs.sampleOfInterest()].raw()&mask));
  val &= (lhs.id().rawId()                  == rhs.id().rawId());
  return val;
}

template <> inline bool 
DEutils<HcalTrigPrimDigiCollection>::de_equal(const cand_type& lhs, const cand_type& rhs) {
  bool val = true;
  unsigned int mask = 0x01ff;
  val &= ((lhs.t0().raw()&mask) == (rhs.t0().raw()&mask));
  val &= (lhs.id().rawId()      == rhs.id().rawId());
  return val;
}

template <> inline bool 
DEutils<L1CaloEmCollection>::de_equal(const cand_type& lhs, const cand_type& rhs) {
  bool val = true;
  val &= (lhs.raw()      == rhs.raw()     );
  val &= (lhs.rctCrate() == rhs.rctCrate());
  val &= (lhs.isolated() == rhs.isolated());
  val &= (lhs.index()    == rhs.index()   );
  //val &= (lhs.bx()       == rhs.bx()      );
  return val;
}

template <> inline bool 
DEutils<L1CaloRegionCollection>::de_equal(const cand_type& lhs, const cand_type& rhs) {
  bool val = true;
  val &= (lhs.et()        == rhs.et()       );
  val &= (lhs.rctCrate()  == rhs.rctCrate() );	
  val &= (lhs.rctRegionIndex() == rhs.rctRegionIndex());
  val &= (lhs.id().isHf() == rhs.id().isHf());  
  if (!lhs.id().isHf()){
    val &= (lhs.overFlow()  == rhs.overFlow() );
    val &= (lhs.tauVeto()   == rhs.tauVeto()  );
    //mask temporarily (!) mip and quiet bits
    //val &= (lhs.mip()       == rhs.mip()      );
    //val &= (lhs.quiet()     == rhs.quiet()    );
    val &= (lhs.rctCard()   == rhs.rctCard()  );
  } else {
    val &= (lhs.fineGrain() == rhs.fineGrain());
  }
  return val;
}

template <> inline bool 
DEutils<L1GctEmCandCollection>::de_equal(const cand_type& lhs, const cand_type& rhs) {return lhs==rhs;}

template <> inline bool  DEutils<L1GctJetCandCollection>::de_equal(const cand_type& lhs, const cand_type& rhs) {return lhs==rhs;}

template <> inline bool  DEutils<L1GctEtHadCollection>::de_equal(const cand_type& lhs, const cand_type& rhs) {return lhs==rhs;}
template <> inline bool DEutils<L1GctEtMissCollection>::de_equal(const cand_type& lhs, const cand_type& rhs) {return lhs==rhs;}
template <> inline bool DEutils<L1GctEtTotalCollection>::de_equal(const cand_type& lhs, const cand_type& rhs) {return lhs==rhs;}
template <> inline bool DEutils<L1GctHtMissCollection>::de_equal(const cand_type& lhs, const cand_type& rhs) {return lhs==rhs;}
template <> inline bool DEutils<L1GctHFRingEtSumsCollection>::de_equal(const cand_type& lhs, const cand_type& rhs) {return lhs==rhs;}
template <> inline bool DEutils<L1GctHFBitCountsCollection>::de_equal(const cand_type& lhs, const cand_type& rhs) {return lhs==rhs;}
template <> inline bool DEutils<L1GctJetCountsCollection>::de_equal(const cand_type& lhs, const cand_type& rhs) {return lhs==rhs;}


template <> inline bool 
DEutils<L1MuDTChambPhDigiCollection>::de_equal(const cand_type& lhs, const cand_type& rhs) {
  bool val = true;
  val &= (lhs.whNum() ==rhs.whNum() );
  val &= (lhs.scNum() ==rhs.scNum() );
  val &= (lhs.stNum() ==rhs.stNum() );
  //val &= (lhs.phi()   ==rhs.phi()   );
  //val &= (lhs.phiB()  ==rhs.phiB()  );
  val &= (lhs.code()  ==rhs.code()  );
  val &= (lhs.Ts2Tag()==rhs.Ts2Tag());
  //val &= (lhs.BxCnt() ==rhs.BxCnt() ); 
  //val &= (lhs.bxNum() ==rhs.bxNum() );
  return val;
}

template <> inline bool 
DEutils<L1MuDTChambThDigiCollection>::de_equal(const cand_type& lhs, const cand_type& rhs) {
  bool val = true;
  val &= (lhs.whNum() ==rhs.whNum() );
  val &= (lhs.scNum() ==rhs.scNum() );
  val &= (lhs.stNum() ==rhs.stNum() );
  //for(int i=0; i<7; i++) {
  //  val &= (lhs.code(i)    ==rhs.code(i)    );
  //  val &= (lhs.position(i)==rhs.position(i));
  //  val &= (lhs.quality(i) ==rhs.quality(i) );
  //}
  //val &= (lhs.bxNum() ==rhs.bxNum() );
  return val;
}

template <> inline bool 
DEutils<L1MuRegionalCandCollection>::de_equal(const cand_type& lhs, const cand_type& rhs) {
  bool val = true;
  val &= (lhs.type_idx() == rhs.type_idx());
  val &= (lhs.bx()       == rhs.bx());
  if(!val) return val;
  unsigned int dw = lhs.getDataWord();
  unsigned int ew = rhs.getDataWord();
  unsigned int mask = 0xffffffff; //32-bit
  //RPC: mask bits 25-29 (including synch bits)
  // emulator doesn't set these bits (permanent masking)
  // 0 DT, 1 bRPC, 2 CSC, 3 fRPC
  if(rhs.type_idx()==1 || rhs.type_idx()==3)
    mask &= 0xc1ffffff;
  dw &= mask; ew &= mask;
  val &= (dw==ew);
  //val &= (lhs.getDataWord() == rhs.getDataWord() );
  //check whether collections being compared refer to same system and bx!
  return val;
}

template <> inline bool 
DEutils<L1MuGMTCandCollection>::de_equal(const cand_type& lhs, const cand_type& rhs) {
  //return (lhs.getDataWord() == rhs.getDataWord() );
  //return lhs==rhs; //(dataword,bx..)
  bool val = true;
  unsigned int dw = rhs.getDataWord();
  unsigned int ew = lhs.getDataWord();
  unsigned int mask = 0x3ffffff; //26-bit
  //mask bits 22 (isolation), 23 (mip) [not permanent!]
  mask &= (~(0x0c00000)); 
  dw &= mask; ew &= mask;
  val &= (dw==ew);
  return val;
  }

template <> inline bool 
DEutils<CSCCorrelatedLCTDigiCollection_>::de_equal(const cand_type& lhs, const cand_type& rhs) {
  // Exclude track number from comparison since it is only meaningful for
  // LCTs upstream of the MPC but not downstream (the latter ones are
  // unpacked by the CSC TF unpacker).
  bool val = true;
  val &= (lhs.isValid()    == rhs.isValid()   );
  val &= (lhs.getQuality() == rhs.getQuality());
  val &= (lhs.getKeyWG()   == rhs.getKeyWG()  );
  val &= (lhs.getStrip()   == rhs.getStrip()  );
  val &= (lhs.getPattern() == rhs.getPattern());
  val &= (lhs.getBend()    == rhs.getBend()   );
  val &= (lhs.getBX()      == rhs.getBX()     );    
  val &= (lhs.getMPCLink() == rhs.getMPCLink()); 
  val &= (lhs.getCSCID()   == rhs.getCSCID()  );
  return val;
  //return lhs==rhs;
}
template <> inline bool 
DEutils<CSCALCTDigiCollection_>::de_equal(const cand_type& lhs, const cand_type& rhs) {
  return lhs==rhs;
}
template <> inline bool 
DEutils<CSCCLCTDigiCollection_>::de_equal(const cand_type& lhs, const cand_type& rhs) {
  return lhs==rhs;
}
template <> inline bool 
DEutils<L1CSCSPStatusDigiCollection_>::de_equal(const cand_type& lhs, const cand_type& rhs) {
  bool val = true;
  val &= (lhs.slot() == rhs.slot());
  val &= (lhs.BXN () == rhs.BXN ());
  val &= (lhs.FMM () == rhs.FMM ());
  val &= (lhs.SEs () == rhs.SEs ());
  val &= (lhs.SMs () == rhs.SMs ());
  val &= (lhs.BXs () == rhs.BXs ());
  val &= (lhs.AFs () == rhs.AFs ());
  val &= (lhs.VPs () == rhs.VPs ());
  return val;
}

/// --- candidate location-match definition ---

template <typename T>
bool DEutils<T>::de_equal_loc(const cand_type& lhs, const cand_type& rhs) {
  //declare candidate matching by default
  return true;
}
template <typename T>
bool DEutils<T>::de_nequal_loc(const cand_type& lhs, const cand_type& rhs) {
  return !de_equal_loc(lhs,rhs);
}


template <> inline bool 
DEutils<EcalTrigPrimDigiCollection>::de_equal_loc(const cand_type& lhs, const cand_type& rhs) {
  bool val = true;
  val &= (lhs.id().zside()   == rhs.id().zside()  );
  val &= (lhs.id().ietaAbs() == rhs.id().ietaAbs());
  val &= (lhs.id().iphi()    == rhs.id().iphi()   );
  return val;
}

template <> inline bool 
DEutils<HcalTrigPrimDigiCollection>::de_equal_loc(const cand_type& lhs, const cand_type& rhs) {
  bool val = true;
  val &= (lhs.id().zside()   == rhs.id().zside()  );
  val &= (lhs.id().ietaAbs() == rhs.id().ietaAbs());
  val &= (lhs.id().iphi()    == rhs.id().iphi()   );
  return val;
}

template <> inline bool 
DEutils<L1CaloEmCollection>::de_equal_loc(const cand_type& lhs, const cand_type& rhs) {
  bool val = true;
  val &= (lhs.rctCrate()  == rhs.rctCrate());
  val &= (lhs.rctCard()   == rhs.rctCard());
  val &= (lhs.rctRegion() == rhs.rctRegion());
  return val;
}

template <> inline bool 
DEutils<L1CaloRegionCollection>::de_equal_loc(const cand_type& lhs, const cand_type& rhs) {
  bool val = true;
  val &= (lhs.rctCrate()  == rhs.rctCrate() );	
  val &= (lhs.id().isHf() == rhs.id().isHf());  
  if (!lhs.id().isHf())
    val &= (lhs.rctCard() == rhs.rctCard()  );
  val &= (lhs.rctRegionIndex() == rhs.rctRegionIndex());
  return val;
}

template <> inline bool 
DEutils<L1GctEmCandCollection>::de_equal_loc(const cand_type& lhs, const cand_type& rhs) {
  bool val = true;
  val &= (lhs.etaIndex() == rhs.etaIndex());
  val &= (lhs.phiIndex() == rhs.phiIndex());
  return val;
}
template <> inline bool 
DEutils<L1GctJetCandCollection>::de_equal_loc(const cand_type& lhs, const cand_type& rhs) {
  bool val = true;
  val &= (lhs.etaIndex() == rhs.etaIndex());
  val &= (lhs.phiIndex() == rhs.phiIndex());
  return val;
}

template <> inline bool 
DEutils<L1GctEtHadCollection>::de_equal_loc(const cand_type& lhs, const cand_type& rhs) {
  return true; // no associated location defined
}
template <> inline bool 
DEutils<L1GctEtMissCollection>::de_equal_loc(const cand_type& lhs, const cand_type& rhs) {
  bool val = true;
  val &= (lhs.phi() == rhs.phi());
  return val;
}
template <> inline bool 
DEutils<L1GctEtTotalCollection>::de_equal_loc(const cand_type& lhs, const cand_type& rhs) {
  return true; // no associated location defined
}
template <> inline bool 
DEutils<L1GctHtMissCollection>::de_equal_loc(const cand_type& lhs, const cand_type& rhs) {
  bool val = true;
  val &= (lhs.phi() == rhs.phi());
  return val;
}
template <> inline bool 
DEutils<L1GctHFRingEtSumsCollection>::de_equal_loc(const cand_type& lhs, const cand_type& rhs) {
  return true; // no associated location defined
}
template <> inline bool 
DEutils<L1GctHFBitCountsCollection>::de_equal_loc(const cand_type& lhs, const cand_type& rhs) {
  return true; // no associated location defined
}
template <> inline bool 
DEutils<L1GctJetCountsCollection>::de_equal_loc(const cand_type& lhs, const cand_type& rhs) {
  return true; // no associated location defined
}


template <> inline bool 
DEutils<L1MuRegionalCandCollection>::de_equal_loc(const cand_type& lhs, const cand_type& rhs) {
  bool val = true;
  val &= (lhs.phi_packed() ==rhs.phi_packed() );
  val &= (lhs.eta_packed() ==rhs.eta_packed() );
  //val &= (lhs.type_idx() == rhs.type_idx());
  //val &= (lhs.bx()       == rhs.bx());
  return val;
}

template <> inline bool 
DEutils<L1MuGMTCandCollection>::de_equal_loc(const cand_type& lhs, const cand_type& rhs) {
  bool val = true;
  val &= (lhs.phiIndex() ==rhs.phiIndex() );
  val &= (lhs.etaIndex() ==rhs.etaIndex() );
  return val;
}

template <> inline bool 
DEutils<L1MuDTChambPhDigiCollection>::de_equal_loc(const cand_type& lhs, const cand_type& rhs) {
  bool val = true;
  val &= (lhs.whNum() ==rhs.whNum() );
  val &= (lhs.scNum() ==rhs.scNum() );
  val &= (lhs.stNum() ==rhs.stNum() );
  //val &= (lhs.phi()   ==rhs.phi()   );
  //val &= (lhs.phiB()  ==rhs.phiB()  );
  //val &= (lhs.bxNum() ==rhs.bxNum() );
  return val;
}

template <> inline bool 
DEutils<L1MuDTChambThDigiCollection>::de_equal_loc(const cand_type& lhs, const cand_type& rhs) {
  bool val = true;
  val &= (lhs.whNum() ==rhs.whNum() );
  val &= (lhs.scNum() ==rhs.scNum() );
  val &= (lhs.stNum() ==rhs.stNum() );
  //val &= (lhs.bxNum() ==rhs.bxNum() );
  return val;
}

template <> inline bool 
DEutils<CSCCorrelatedLCTDigiCollection_>::de_equal_loc(const cand_type& lhs, const cand_type& rhs) {
  bool val = true;
  val &= (lhs.getCSCID() == rhs.getCSCID() );
  val &= (lhs.getStrip() == rhs.getStrip() );
  val &= (lhs.getKeyWG() == rhs.getKeyWG() );
  return val;
}

template <> inline bool 
DEutils<CSCALCTDigiCollection_>::de_equal_loc(const cand_type& lhs, const cand_type& rhs) {
  bool val = true;
  val &= (lhs.getTrknmb() == rhs.getTrknmb() );
  val &= (lhs.getKeyWG()  == rhs.getKeyWG()  );
  return val;
}
template <> inline bool 
DEutils<CSCCLCTDigiCollection_>::de_equal_loc(const cand_type& lhs, const cand_type& rhs) {
  bool val = true;
  val &= (lhs.getTrknmb()   == rhs.getTrknmb()   );
  val &= (lhs.getKeyStrip() == rhs.getKeyStrip() );
  return val;
}
template <> inline bool 
DEutils<L1CSCSPStatusDigiCollection_>::de_equal_loc(const cand_type& lhs, const cand_type& rhs) {
  bool val = true;
  val &= (lhs.slot() == rhs.slot());
  return val;
}
 
/// --- candidate emptiness definition ---

template <typename T> 
bool DEutils<T>::is_empty(col_cit it) const { 
  //declare candidate non-empty by default
  return false; 
}

template<>
inline bool DEutils<EcalTrigPrimDigiCollection>::is_empty(col_cit it) const { 
  bool val = false;
  unsigned int raw = it->sample(it->sampleOfInterest()).raw();
  unsigned int mask = 0x0fff;
  mask = 0x0eff; //fg bit temporary(!) mask
  raw &= mask;
  val |= (raw==0);
  if(val) return val;
  unsigned int ttf = it->ttFlag();
  val |= ((ttf!=0x1) && (ttf!=0x3)); //compare only if ttf is 1 or 3
  return val;  
  //  return ( it->size()==0 || it->sample(it->sampleOfInterest()).raw()==0);
}

template<>
inline bool DEutils<HcalTrigPrimDigiCollection>::is_empty(col_cit it) const { 
  unsigned int mask = 0x01ff;
  return (  it->size()==0 || ((it->t0().raw()&mask)==0) || it->SOI_compressedEt()==0 );
}

template<>
inline bool DEutils<L1CaloEmCollection>::is_empty(col_cit it) const { 
  return  ((it->rank())==0);
  //return it->empty();
}

template<>
inline bool DEutils<L1CaloRegionCollection>::is_empty(col_cit it) const { 
  return  ((it->et())==0);
  //return it->empty();
}

template<>
inline bool DEutils<L1GctEmCandCollection>::is_empty(col_cit it) const { 
  return (it->empty());
}

template<>
inline bool DEutils<L1GctJetCandCollection>::is_empty(col_cit it) const { 
    return  (it->empty());
}

template <> inline bool DEutils<L1GctEtHadCollection>::is_empty(col_cit it) const {return(it->empty());}
template <> inline bool DEutils<L1GctEtMissCollection>::is_empty(col_cit it) const {return(it->empty());}
template <> inline bool DEutils<L1GctEtTotalCollection>::is_empty(col_cit it) const {return(it->empty());}
template <> inline bool DEutils<L1GctHtMissCollection>::is_empty(col_cit it) const {return(it->empty());}
template <> inline bool DEutils<L1GctHFRingEtSumsCollection>::is_empty(col_cit it) const {return(it->empty());}
template <> inline bool DEutils<L1GctHFBitCountsCollection>::is_empty(col_cit it) const {return(it->empty());}
template <> inline bool DEutils<L1GctJetCountsCollection>::is_empty(col_cit it) const {return(it->empty());}


template<>
inline bool DEutils<L1MuDTChambPhDigiCollection>::is_empty(col_cit it) const { 
  return (it->bxNum() != 0 || it->code() == 7); 
  //return (it->qualityCode() == 7); 
  //return  false;
}
template<>
inline bool DEutils<L1MuDTChambThDigiCollection>::is_empty(col_cit it) const { 
  return (it->whNum()==0 && it->scNum()==0 && it->stNum()==0);//tmp!
  //return  false;
}

template<>
inline bool DEutils<L1MuRegionalCandCollection>::is_empty(col_cit it) const { 
  //note: following call used to give trouble sometimes
  
  //restrict further processing to bx==0 for RPC 
  if(it->type_idx()==1 || it->type_idx()==3) //rpc
    if (it->bx()!=0) 
      return true; 
  
  return (it->empty()); 
  //virtual bool empty() const { return readDataField( PT_START, PT_LENGTH) == 0; }
  //return  (it->getDataWord()==0);
  //return  (it->pt_packed()==0);
}

template<>
inline bool DEutils<L1MuGMTCandCollection>::is_empty(col_cit it) const { 
  return (it->empty());
  //return (it->ptIndex()==0);
  //return  (it->getDataWord()==0);
}

template<>
inline bool DEutils<CSCCorrelatedLCTDigiCollection_>::is_empty(col_cit it) const { 
  return !(it->isValid());
}
template<>
inline bool DEutils<CSCALCTDigiCollection_>::is_empty(col_cit it) const { 
  return !(it->isValid());
}
template<>
inline bool DEutils<CSCCLCTDigiCollection_>::is_empty(col_cit it) const { 
  return !(it->isValid());
}

template<>
inline bool DEutils<L1CSCSPStatusDigiCollection_>::is_empty(col_cit it) const { 
  unsigned data = 
    it->slot() | it->BXN () | it->FMM () | it->SEs () |
    it->SMs () | it->BXs () | it->AFs () | it->VPs ();
  return data==0;
}

/// --- print candidate ---

template <typename T> 
std::string DEutils<T>::print(col_cit it) const {
  std::stringstream ss;
  ss << "[DEutils<T>::print()] specialization still missing for collection!";
  //ss << *it; // default
  ss << std::endl;
  return ss.str();
}

template <> 
inline std::string DEutils<EcalTrigPrimDigiCollection>::print(col_cit it) const {
  std::stringstream ss;
  ss << "0x" << std::setw(4) << std::setfill('0') << std::hex 
     << it->sample(it->sampleOfInterest()).raw()
     << std::setfill(' ') << std::dec 
     << ", et:"   << std::setw(3) << it->compressedEt() 
     << ", fg:"   << std::setw(1) << it->fineGrain()
     << ", ttf:"  << std::setw(2) << it->ttFlag()
     << ", sdet:" << ((it->id().subDet()==EcalBarrel)?("Barrel"):("Endcap")) 
     << ", iz:"   << ((it->id().zside()>0)?("+"):("-")) 
     << ", ieta:" << std::setw(2) << it->id().ietaAbs()
     << ", iphi:" << std::setw(2) << it->id().iphi()
    //<< "\n\t: " << *it 
     << std::endl;
  return ss.str();
}

template <> 
inline std::string DEutils<HcalTrigPrimDigiCollection>::print(col_cit it) const {
  std::stringstream ss;
  ss << "0x" << std::setw(4) << std::setfill('0') << std::hex 
     << it->t0().raw()
     << std::setfill(' ') << std::dec 
     << ", et:"   << std::setw(3) << it->SOI_compressedEt()
     << ", fg:"   << std::setw(1) << it->SOI_fineGrain()
     << ", sdet:" << it->id().subdet()
     << ", iz:"   << ((it->id().zside()>0)?("+"):("-")) 
     << ", ieta:" << std::setw(2) << it->id().ietaAbs()
     << ", iphi:" << std::setw(2) << it->id().iphi()
     << std::endl;
  //ss << *it << std::endl;
  return ss.str();
}

template <> 
inline std::string DEutils<L1CaloEmCollection>::print(col_cit it) const {
  std::stringstream ss;
  ss << "0x" << std::setw(4) << std::setfill('0') << std::hex << it->raw() 
     << ", rank=0x"<< std::setw(2) << std::hex << it->rank()
     << std::setfill(' ') << std::dec 
     << ", region:"<< std::setw(1) << it->rctRegion() 
     << ", card:"  << std::setw(1) << it->rctCard() 
     << ", crate:" << std::setw(2) << it->rctCrate()
     << ", ieta:"  << std::setw(2) << it->regionId().ieta() //0..21
     << ", iphi:"  << std::setw(2) << it->regionId().iphi() //0..17
    //<< ", eta:"   << std::setw(2) << it->regionId().rctEta() //0..10
    //<< ", phi:"   << std::setw(2) << it->regionId().rctPhi() //0..1
     << ", iso:"   << std::setw(1) << it->isolated()
     << ", index:" << std::setw(1) << it->index() 
     << ", bx:"    << it->bx()
     << std::endl;
  //ss << *it;
  return ss.str();
}

template <> 
inline std::string DEutils<L1CaloRegionCollection>::print(col_cit it) const {
  std::stringstream ss;
  ss << "L1CaloRegion:"
     << " et=" << it->et()
     << " o/f=" << it->overFlow()
     << " f/g=" << it->fineGrain()
     << " tau=" << it->tauVeto()
     << " rct(crate=" << it->rctCrate()
     << " card=" << it->rctCard()
     << " rgn=" << it->rctRegionIndex()
     << " eta=" << it->rctEta()
     << " phi=" << it->rctPhi()
     << ")\n\t\t"
     << "gct(eta=" << it->gctEta()
     << " phi=" << it->gctPhi()
     << ")"
     << std::hex << " cap_block=" << it->capBlock() 
     << std::dec << " index=" << it->capIndex()
     << " bx=" << it->bx()
     << std::endl;
  //ss << *it; ///replace due to too long unformatted verbose
  //note: raw() data accessor missing in dataformats
  return ss.str();
}

template <> 
inline std::string DEutils<L1GctEmCandCollection>::print(col_cit it) const {
  std::stringstream ss;
  //get rct index
  //int ieta = (it->etaIndex()&0x7); ieta = it->etaSign() ? 10-ieta:11+ieta; 
  ss << "0x" << std::setw(4) << std::setfill('0') << std::hex << it->raw() 
     << ", rank=0x"<< std::setw(2) << std::hex << it->rank()
     << std::setfill(' ') << std::dec 
     << ", etaSign:"   <<  it->etaSign() 
     << ", eta:"       << (it->etaIndex()&0x7)           //0..6
     << ", phi:"      << std::setw(2) << it->phiIndex()  //0..17
     << " (ieta:" << std::setw(2) << it->regionId().ieta() //0..21
     << ",iphi:"  << std::setw(2) << it->regionId().iphi() << ")" //0..17
     << ", iso:"       <<  it->isolated()
     << ", cap block:" << std::setw(3) << it->capBlock() 
     << ", index:"     <<  it->capIndex() 
     << ", bx:"        <<  it->bx()
     << std::endl;
  //<< " " << *it << std::dec << std::endl;
  return ss.str();
}

/*notes on rct/gct indices
 ieta: 0 .. 11 .. 21  ->  rctEta: 10 .. 0 .. 10   ie: ieta<11?10-ieta:ieta-11
 gct from rct  eta: rctEta&0x7 (7..0..7) | etaSign(==ieta<11)
 rct from gct  eta: +-(0..7) -> 3..18  ie: sign?10-eta:eta+11
 rct iphi = gct phi
*/

template <> 
inline std::string DEutils<L1GctJetCandCollection>::print(col_cit it) const {
  std::stringstream ss;
  ss << "0x" << std::setw(4) << std::setfill('0') << std::hex << it->raw() 
     << " " << *it << std::dec << std::endl; 
  return ss.str();
}
template <> inline std::string DEutils<L1GctEtHadCollection	   >::print(col_cit it) const {std::stringstream ss; ss<<*it; return ss.str();}
template <> inline std::string DEutils<L1GctEtMissCollection	   >::print(col_cit it) const {std::stringstream ss; ss<<*it; return ss.str();}
template <> inline std::string DEutils<L1GctEtTotalCollection	   >::print(col_cit it) const {std::stringstream ss; ss<<*it; return ss.str();}
template <> inline std::string DEutils<L1GctHtMissCollection	   >::print(col_cit it) const {std::stringstream ss; ss<<*it; return ss.str();}
template <> inline std::string DEutils<L1GctHFRingEtSumsCollection>::print(col_cit it) const {std::stringstream ss; ss<<*it; return ss.str();}
template <> inline std::string DEutils<L1GctHFBitCountsCollection >::print(col_cit it) const {std::stringstream ss; ss<<*it; return ss.str();}
template <> inline std::string DEutils<L1GctJetCountsCollection   >::print(col_cit it) const {std::stringstream ss; ss<<*it; return ss.str();}

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

template <> 
inline std::string DEutils<L1MuRegionalCandCollection>::print(col_cit it) const {
  std::stringstream ss;
  //const float noval = -10; //L1MuRegionalCand::m_invalidValue;
  ss << std::setiosflags(std::ios::showpoint | std::ios::fixed | std::ios::right | std::ios::adjustfield);
  ss   << std::hex << std::setfill('0')    
       << " 0x"    << std::setw(8) << it->getDataWord();
  //if(it->phiValue()==noval || it->etaValue()==noval || it->ptValue()==noval )
    ss << std::hex << std::setfill('0')    
       << " pt:0x" << std::setw(2) << it->pt_packed() 
       << " phi:0x"<< std::setw(2) << it->phi_packed()
       << " eta:0x"<< std::setw(2) << it->eta_packed();
  //else
  //  ss << std::dec << std::setfill(' ')
  //     << " pt:"   << std::setw(5) << std::setprecision(1) << it->ptValue() <<"[GeV]"
  //     << " phi:"  << std::setw(5) << std::setprecision(3) << it->phiValue()<<"[rad]"
  //     << " eta:"  << std::setw(6) << std::setprecision(3) << it->etaValue(); 
  ss   << std::dec << std::setfill(' ')
       << " qua:"  << std::setw(1) << it->quality() 
       << " cha:"  << std::setw(2) << it->chargeValue() 
       << " chav:" << std::setw(1) << it->chargeValid() 
       << " fh:"   << std::setw(1) << it->isFineHalo() 
       << " bx:"   << std::setw(4) << it->bx() 
       << " [id:"  << std::setw(1) << it->type_idx() << "]" // 0 DT, 1 bRPC, 2 CSC, 3 fRPC
       << std::endl;  
  //ss << it->print() 
  return ss.str();
}

template <> 
inline std::string DEutils<L1MuGMTCandCollection>::print(col_cit it) const {
  std::stringstream ss;
  ss << std::setiosflags(std::ios::showpoint | std::ios::fixed | std::ios::right | std::ios::adjustfield);
  //const float noval = -10; //L1MuGMTCand::m_invalidValue;
  ss   << std::hex << std::setfill('0')    
       << " 0x"    << std::setw(7) << it->getDataWord();
  //if(it->phiValue()==noval || it->etaValue()==noval || it->ptValue()==noval)
    ss << std::hex << std::setfill('0')    
       << " pt:0x" << std::setw(2) << it->ptIndex()
       << " eta:0x"<< std::setw(2) << it->etaIndex()
       << " phi:0x"<< std::setw(3) << it->phiIndex();
  //else
  //  ss << std::dec << std::setfill(' ')    
  //     << " pt:"   << std::setw(5) << std::setprecision(1) << it->ptValue() <<"[GeV]"
  //     << " phi:"  << std::setw(5) << std::setprecision(3) << it->phiValue()<<"[rad]"
  //     << " eta:"  << std::setw(6) << std::setprecision(2) << it->etaValue();
  ss   << std::dec << std::setfill(' ')
       << " cha:"  << std::setw(2) << it->charge()  
       << " qua:"  << std::setw(3) << it->quality() 
       << " iso:"  << std::setw(1) << it->isol()    
       << " mip:"  << std::setw(1) << it->mip() 
       << " bx:"                   << it->bx() 
       << std::endl;
  //ss << it->print() 
  return ss.str();
}

template <> 
inline std::string DEutils<CSCCorrelatedLCTDigiCollection_>::print(col_cit it) const {
  std::stringstream ss;
  ss 
    //<< " lct#:"     << it->getTrknmb()
    //<< " val:"      << it->isValid()
    //<< " qua:"      << it->getQuality() 
    //<< " strip:"    << it->getStrip()
    //<< " bend:"     << ((it->getBend() == 0) ? 'L' : 'R')
    //<< " patt:"     << it->getPattern()
    //<<"  key wire:" << it->getKeyWG()
    //<< " bx:"       << it->getBX()
    //<< " mpc-link:" << it->getMPCLink()
    //<< " csc id:"   << it->getCSCID()
    //<< std::endl;
    << *it;
  return ss.str();
}

template <> 
inline std::string DEutils<CSCALCTDigiCollection_>::print(col_cit it) const {
  std::stringstream ss;
  ss
    << *it
    << std::endl;
  return ss.str();
}

template <> 
inline std::string DEutils<CSCCLCTDigiCollection_>::print(col_cit it) const {
  std::stringstream ss;
  ss 
    << *it
    << std::endl;
  return ss.str();
}

template <> 
inline std::string DEutils<L1CSCSPStatusDigiCollection_>::print(col_cit it) const {
  std::stringstream ss;
  ss 
    << " slot:"<< it->slot()
    << " bxn:" << it->BXN ()
    << " fmm:" << it->FMM ()
    << " ses:" << it->SEs ()
    << " sms:" << it->SMs ()
    << " bxs:" << it->BXs ()
    << " afs:" << it->AFs ()
    << " vps:" << it->VPs ()
    << std::endl;
  return ss.str();
}

/// --- name candidate ---

template <typename T> 
std::string DEutils<T>::GetName(int i) const {

  const int nlabel = 17;
  if(!(i<nlabel)) 
    return                  "un-defined" ;
  std::string str[nlabel]= {"un-registered"};

  switch(de_type()) {
  case dedefs::ECALtp:
    str[0] = "ECAL tp";
    str[1] = "EcalTrigPrimDigiCollection";
    str[2] = "EcalTriggerPrimitiveDigi";
  break;
  case dedefs::HCALtp:
    str[0] = "HCAL tp";
    str[1] = "HcalTrigPrimDigiCollection";
    str[2] = "HcalTriggerPrimitiveDigi";
  break;
  case dedefs::RCTem:
    str[0] = "RCT em";
    str[1] = "L1CaloEmCollection";
    str[2] = "L1CaloEmCand";
  break;
  case dedefs::RCTrgn:
    str[0] = "RCT region";
    str[1] = "L1CaloRegionCollection";
    str[2] = "L1CaloRegion";
    break;
  case dedefs::GCTisolaem:
    str[0] = "GCT em isolated";
    str[1] = "L1GctEmCandCollection";
    str[2] = "L1GctEmCand";
   break;
  case dedefs::GCTnoisoem:
    str[0] = "GCT em non-isolated";
    str[1] = "L1GctEmCandCollection";
    str[2] = "L1GctEmCand";
   break;
  case dedefs::GCTcenjets:
    str[0] = "GCT central jet";
    str[1] = "L1GctJetCandCollection";
    str[2] = "L1GctJetCand";
   break;
  case dedefs::GCTforjets:
    str[0] = "GCT forward jet";
    str[1] = "L1GctJetCandCollection";
    str[2] = "L1GctJetCand";
   break;
  case dedefs::GCTtaujets:
    str[0] = "GCT tau jet";
    str[1] = "L1GctJetCandCollection";
    str[2] = "L1GctJetCand";
   break;
  case dedefs::GCTisotaujets:
    str[0] = "Stage1Layer2 iso-tau jet";
    str[1] = "L1GctJetCandCollection";
    str[2] = "L1GctJetCand";
   break;
  case dedefs::GCTethad:
    str[0] = "GCT ht";
    str[1] = "L1GctEtHadCollection";
    str[2] = "L1GctEtHad";
   break;
  case dedefs::GCTetmiss:
    str[0] = "GCT et miss";
    str[1] = "L1GctEtMissCollection";
    str[2] = "L1GctEtMiss";
   break;
  case dedefs::GCTettot:
    str[0] = "GCT et total";
    str[1] = "L1GctEtTotalCollection";
    str[2] = "L1GctEtTotal";
   break;
  case dedefs::GCThtmiss:
    str[0] = "GCT ht miss";
    str[1] = "L1GctHtMissCollection";
    str[2] = "L1GctHtMiss";
   break;
  case dedefs::GCThfring:
    str[0] = "GCT hf ring";
    str[1] = "L1GctHFRingEtSumsCollection";
    str[2] = "L1GctHFRingEtSums";
   break;
  case dedefs::GCThfbit:
    str[0] = "GCT hf bit counts";
    str[1] = "L1GctHFBitCountsCollection";
    str[2] = "L1GctHFBitCounts";
   break;

  case dedefs::DTtpPh:
    str[0] = "DT tp phi";
    str[1] = "L1MuDTChambPhDigiCollection";
    str[2] = "L1MuDTChambPhDigi";
   break;
  case dedefs::DTtpTh:
    str[0] = "DT tp theta";
    str[1] = "L1MuDTChambThDigiCollection";
    str[2] = "L1MuDTChambThDigi";
   break;
  case dedefs::CSCtpa:
    str[0] = "CSC tpa";
    str[1] = "CSCALCTDigiCollection";
    str[2] = "CSCALCTDigi";
   break;
  case dedefs::CSCtpc:
    str[0] = "CSC tpc";
    str[1] = "CSCCLCTDigiCollection";
    str[2] = "CSCCLCTDigi";
   break;
  case dedefs::CSCtpl:
    str[0] = "CSC tp";
    str[1] = "CSCCorrelatedLCTDigiCollection";
    str[2] = "CSCCorrelatedLCTDigi";
   break;
  case dedefs::CSCsta:
    str[0] = "CSC tf status";
    str[1] = "L1CSCSPStatusDigiCollection_";
    str[2] = "L1CSCSPStatusDigi";
   break;
  case dedefs::MUrtf:
    str[0] = "Mu reg tf";
    str[1] = "L1MuRegionalCandCollection";
    str[2] = "L1MuRegionalCand";
   break;
  case dedefs::LTCi:
    str[0] = "LTC";
    str[1] = "LTCDigiCollection";
    str[2] = "LTCDigi";
    break;
  case dedefs::GMTcnd:
    str[0] = "GMT cand";
    str[1] = "L1MuGMTCandCollection";
    str[2] = "L1MuGMTCand";
    break;
  case dedefs::GMTrdt:
    str[0] = "GMT record";
    str[1] = "L1MuGMTReadoutRecordCollection";
    str[2] = "L1MuGMTReadoutRecord";
    break;
  case dedefs::GTdword:
    str[0] = "";
    str[1] = "";
    str[2] = "";
    break;
    //default:
  }
  return str[i];
}

/// --- order candidates ---

template <typename T>
struct de_rank : public DEutils<T> , public std::binary_function<typename DEutils<T>::cand_type, typename DEutils<T>::cand_type, bool> {
  typedef DEtrait<T> de_trait;
  typedef typename de_trait::cand_type cand_type;
  bool operator()(const cand_type& x, const cand_type& y) const {
    return false; //default
  }
};

template <> inline bool de_rank<EcalTrigPrimDigiCollection>::operator()(const cand_type& x, const cand_type& y) const { return x.compressedEt() > y.compressedEt(); }
template <> inline bool de_rank<HcalTrigPrimDigiCollection>::operator()(const cand_type& x, const cand_type& y) const { return x.SOI_compressedEt() > y.SOI_compressedEt(); }

template <> 
inline bool de_rank<L1CaloEmCollection>::operator() 
     (const cand_type& x, const cand_type& y) const {
  if       (x.rank()      != y.rank())     {
    return (x.rank()      <  y.rank())     ;
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

template <> inline bool de_rank<L1CaloRegionCollection>::operator()(const cand_type& x, const cand_type& y) const { return x.et() < y.et(); }

template <> inline bool de_rank<L1GctEmCandCollection>::operator()(const cand_type& x, const cand_type& y)const { if(x.rank()!=y.rank()){return x.rank() < y.rank();} else{if(x.etaIndex()!=y.etaIndex()){return y.etaIndex() < x.etaIndex();}else{ return x.phiIndex() < y.phiIndex();}}}
template <> inline bool de_rank<L1GctJetCandCollection>::operator()(const cand_type& x, const cand_type& y)const { if(x.rank()!=y.rank()){return x.rank() < y.rank();} else{if(x.etaIndex()!=y.etaIndex()){return y.etaIndex() < x.etaIndex();}else{ return x.phiIndex() < y.phiIndex();}}}
//template <> inline bool de_rank<L1GctEtHadCollection>::operator()(const cand_type& x, const cand_type& y)const { }
//template <> inline bool de_rank<L1GctEtMissCollection>::operator()(const cand_type& x, const cand_type& y)const { }
//template <> inline bool de_rank<L1GctEtTotalCollection>::operator()(const cand_type& x, const cand_type& y)const { }
//template <> inline bool de_rank<L1GctHtMissCollection>::operator()(const cand_type& x, const cand_type& y)const { }
//template <> inline bool de_rank<L1GctHFRingEtSumsCollection>::operator()(const cand_type& x, const cand_type& y)const { }
//template <> inline bool de_rank<L1GctHFBitCountsCollection>::operator()(const cand_type& x, const cand_type& y)const { }
//template <> inline bool de_rank<L1GctJetCountsCollection>::operator()(const cand_type& x, const cand_type& y)const { }

template <> inline bool de_rank<L1MuDTChambPhDigiCollection>::operator()(const cand_type& x, const cand_type& y)const { if(x.whNum()!=y.whNum()){return x.whNum() < y.whNum();} else{if(x.scNum()!=y.scNum()){return y.scNum() < x.scNum();}else{ return x.stNum() < y.stNum();}}}
template <> inline bool de_rank<L1MuDTChambThDigiCollection>::operator()(const cand_type& x, const cand_type& y)const { if(x.whNum()!=y.whNum()){return x.whNum() < y.whNum();} else{if(x.scNum()!=y.scNum()){return y.scNum() < x.scNum();}else{ return x.stNum() < y.stNum();}}}

template <> inline bool de_rank<L1MuRegionalCandCollection>::operator()(const cand_type& x, const cand_type& y)const {if(x.phi_packed()!=y.phi_packed()){return x.phi_packed() < y.phi_packed();} else{if(x.eta_packed()!=y.eta_packed()){return y.eta_packed() < x.eta_packed();}else{ return x.quality_packed() < y.quality_packed();}}}

template <> inline bool de_rank<L1MuGMTCandCollection>::operator()(const cand_type& x, const cand_type& y)const {
  if(x.bx()!=y.bx()){return x.bx() < y.bx();} 
  else if(x.ptIndex()!=y.ptIndex()){return x.ptIndex() < y.ptIndex();}
  else{ return x.quality() < y.quality();}
}

template <> inline bool de_rank<CSCCorrelatedLCTDigiCollection_>::operator()(const cand_type& x, const cand_type& y)const {if(x.getTrknmb()!=y.getTrknmb()){return x.getTrknmb() < y.getTrknmb();} else{if(x.getKeyWG()!=y.getKeyWG()){return y.getKeyWG() < x.getKeyWG();} else{ return x.getQuality() < y.getQuality();}}}

#endif
