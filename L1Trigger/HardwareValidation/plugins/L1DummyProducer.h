#ifndef L1_DUMMY_PRODUCER_H
#define L1_DUMMY_PRODUCER_H

/*\class L1DummyProducer
 *\description produces simplified, random L1 trigger digis
 *\usage pattern and monitoring software test and validation
 *\author Nuno Leonardo (CERN)
 *\date 07.07
 */

// system includes
#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include "TMath.h"
#include <bitset>
#include <atomic>

// common includes
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// l1 dataformats, d|e record includes
#include "DataFormats/HcalDigi/interface/HcalTriggerPrimitiveDigi.h"
#include "DataFormats/L1CSCTrackFinder/interface/L1Track.h"
#include "L1Trigger/HardwareValidation/interface/DEtrait.h"

// random # generator
#include "CLHEP/Random/RandomEngine.h"
#include "CLHEP/Random/RandGaussQ.h"

class L1DummyProducer : public edm::global::EDProducer<> {
public:
  explicit L1DummyProducer(const edm::ParameterSet&);
  ~L1DummyProducer() override;

private:
  //virtual void beginRun(edm::Run&, const edm::EventSetup&);
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

public:
  template <class T>
  void SimpleDigi(int nevt, CLHEP::HepRandomEngine*, std::unique_ptr<T>& data, int type = 0) const;

private:
  int verbose_;
  int verbose() const { return verbose_; }
  mutable std::atomic<int> nevt_;

  bool m_doSys[dedefs::DEnsys];
  std::string instName[dedefs::DEnsys][5];

  double EBase_;
  double ESigm_;
};

template <class T>
void L1DummyProducer::SimpleDigi(int, CLHEP::HepRandomEngine*, std::unique_ptr<T>& data, int type) const {
  /*collections generated in specializations below*/
}

template <>
inline void L1DummyProducer::SimpleDigi(int,
                                        CLHEP::HepRandomEngine* engine,
                                        std::unique_ptr<EcalTrigPrimDigiCollection>& data,
                                        int type) const {
  if (verbose())
    std::cout << "L1DummyProducer::SimpleDigi<EcalTrigPrimDigiCollection>....\n" << std::flush;
  int side = (engine->flat() > 0.5) ? -1 : 1;
  int ieta = (int)(1 + 17 * engine->flat());  //1-17
  int iphi = (int)(1 + 72 * engine->flat());  //1-72
  const EcalTrigTowerDetId e_id(side, EcalBarrel, ieta, iphi, 0);
  EcalTriggerPrimitiveDigi e_digi(e_id);
  int energy = (int)(EBase_ + ESigm_ * CLHEP::RandGaussQ::shoot(engine));
  bool fg = (engine->flat() > 0.5);
  int ttf = (int)(8 * engine->flat());  //0-7
  EcalTriggerPrimitiveSample e_sample(energy, fg, ttf);
  e_digi.setSize(1);  //set sampleOfInterest to 0
  e_digi.setSample(0, e_sample);
  data->push_back(e_digi);
  //EcalTriggerPrimitiveSample(int encodedEt, bool finegrain, int triggerFlag);
  //const EcalTrigTowerDetId e_id( zside , EcalBarrel, etaTT, phiTT, 0);
  if (verbose())
    std::cout << "L1DummyProducer::SimpleDigi<EcalTrigPrimDigiCollection> end.\n" << std::flush;
}

template <>
inline void L1DummyProducer::SimpleDigi(int,
                                        CLHEP::HepRandomEngine* engine,
                                        std::unique_ptr<HcalTrigPrimDigiCollection>& data,
                                        int type) const {
  if (verbose())
    std::cout << "L1DummyProducer::SimpleDigi<HcalTrigPrimDigiCollection>....\n" << std::flush;
  int side = (engine->flat() > 0.5) ? -1 : 1;
  int ieta = (int)(1 + 17 * engine->flat());
  int iphi = (int)(1 + 72 * engine->flat());
  const HcalTrigTowerDetId h_id(side * ieta, iphi);
  HcalTriggerPrimitiveDigi h_digi(h_id);
  int energy = (int)(EBase_ + ESigm_ * CLHEP::RandGaussQ::shoot(engine));
  HcalTriggerPrimitiveSample h_sample(energy, false, 0, 0);
  h_digi.setSize(1);  //set sampleOfInterest to 0
  h_digi.setSample(0, h_sample);
  data->push_back(h_digi);
  //HcalTriggerPrimitiveSample(int encodedEt, bool finegrain, int slb, int slbchan);
  //HcalTrigTowerDetId(int ieta, int iphi);
  if (verbose())
    std::cout << "L1DummyProducer::SimpleDigi<HcalTrigPrimDigiCollection> end.\n" << std::flush;
}

template <>
inline void L1DummyProducer::SimpleDigi(int nevt,
                                        CLHEP::HepRandomEngine* engine,
                                        std::unique_ptr<L1CaloEmCollection>& data,
                                        int type) const {
  if (verbose())
    std::cout << "L1DummyProducer::SimpleDigi<L1CaloEmCollection>....\n" << std::flush;
  int energy = (int)(EBase_ + ESigm_ * CLHEP::RandGaussQ::shoot(engine));
  unsigned rank = energy & 0x3f;
  unsigned region = (engine->flat() > 0.5 ? 0 : 1);
  unsigned card = (unsigned)(7 * engine->flat());
  unsigned crate = (unsigned)(18 * engine->flat());
  bool iso = (engine->flat() > 0.4);
  uint16_t index = (unsigned)(4 * engine->flat());
  int16_t bx = nevt;
  L1CaloEmCand cand(rank, region, card, crate, iso, index, bx);
  data->push_back(cand);
  //L1CaloEmCand(unsigned rank, unsigned region, unsigned card, unsigned crate, bool iso, uint16_t index, int16_t bx);
  if (verbose())
    std::cout << "L1DummyProducer::SimpleDigi<L1CaloEmCollection> end.\n" << std::flush;
}

template <>
inline void L1DummyProducer::SimpleDigi(int,
                                        CLHEP::HepRandomEngine* engine,
                                        std::unique_ptr<L1CaloRegionCollection>& data,
                                        int type) const {
  if (verbose())
    std::cout << "L1DummyProducer::SimpleDigi<L1CaloRegionCollection>....\n" << std::flush;
  int energy = (int)(EBase_ + ESigm_ * CLHEP::RandGaussQ::shoot(engine));
  unsigned et = energy & 0x3ff;
  bool overFlow = false;  //(engine->flat()>0.4);
  bool tauVeto = false;   //(engine->flat()>0.3);
  bool mip = false;       //(engine->flat()>0.1);
  bool quiet = false;     //(engine->flat()>0.6);
  unsigned crate = (unsigned)(18 * engine->flat());
  unsigned card = (unsigned)(7 * engine->flat());
  unsigned rgn = crate % 2;  //(engine->flat()>0.5?0:1);
  L1CaloRegion cand(et, overFlow, tauVeto, mip, quiet, crate, card, rgn);
  data->push_back(cand);
  //L1CaloRegion(unsigned et, bool overFlow, bool tauVeto, bool mip, bool quiet, unsigned crate, unsigned card, unsigned rgn);
  if (verbose())
    std::cout << "L1DummyProducer::SimpleDigi<L1CaloRegionCollection> end.\n" << std::flush;
}

template <>
inline void L1DummyProducer::SimpleDigi(int,
                                        CLHEP::HepRandomEngine* engine,
                                        std::unique_ptr<L1GctEmCandCollection>& data,
                                        int type) const {
  if (verbose())
    std::cout << "L1DummyProducer::SimpleDigi<L1GctEmCandCollection>....\n" << std::flush;
  bool iso;        //= type==0;
  switch (type) {  // 0 iso, 1 noniso
    case 0:
      iso = true;
      break;
    case 1:
      iso = false;
      break;
    default:
      throw cms::Exception("L1DummyProducerInvalidType")
          << "L1DummyProducer::SimpleDigi production of L1GctEmCandCollection "
          << " invalid type: " << type << std::endl;
  }
  int energy = (int)(EBase_ + ESigm_ * CLHEP::RandGaussQ::shoot(engine));
  unsigned rank = energy & 0x3f;
  unsigned phi = (unsigned)(18 * engine->flat());
  unsigned eta = (unsigned)(7 * engine->flat());
  if (engine->flat() > 0.5)  //-z (eta sign)
    eta = (eta & 0x7) + (0x1 << 3);
  L1GctEmCand cand(rank, phi, eta, iso);
  data->push_back(cand);
  // eta = -6 to -0, +0 to +6. Sign is bit 3, 1 means -ve Z, 0 means +ve Z
  //L1GctEmCand(unsigned rank, unsigned phi, unsigned eta, bool iso);
  if (verbose())
    std::cout << "L1DummyProducer::SimpleDigi<L1GctEmCandCollection> end.\n" << std::flush;
}

template <>
inline void L1DummyProducer::SimpleDigi(int,
                                        CLHEP::HepRandomEngine* engine,
                                        std::unique_ptr<L1GctJetCandCollection>& data,
                                        int type) const {
  if (verbose())
    std::cout << "L1DummyProducer::SimpleDigi<L1GctJetCandCollection>....\n" << std::flush;
  bool isFor, isTau;
  switch (type) {  // 0 cen, 1 for, 2 tau
    case 0:
      isFor = false;
      isTau = false;
      break;
    case 1:
      isFor = true;
      isTau = false;
      break;
    case 2:
      isFor = false;
      isTau = true;
      break;
    default:
      throw cms::Exception("L1DummyProducerInvalidType")
          << "L1DummyProducer::SimpleDigi production of L1GctJetCandCollection "
          << " invalid type: " << type << std::endl;
  }

  int energy = (int)(EBase_ + ESigm_ * CLHEP::RandGaussQ::shoot(engine));
  unsigned rank = energy & 0x3f;
  unsigned phi = (unsigned)(18 * engine->flat());
  unsigned eta = (unsigned)(7 * engine->flat());
  if (engine->flat() > 0.5)  //-z (eta sign)
    eta = (eta & 0x7) + (0x1 << 3);
  L1GctJetCand cand(rank, phi, eta, isTau, isFor);
  data->push_back(cand);
  //L1GctJetCand(unsigned rank, unsigned phi, unsigned eta, bool isTau, bool isFor);
  if (verbose())
    std::cout << "L1DummyProducer::SimpleDigi<L1GctJetCandCollection> end.\n" << std::flush;
}

template <>
inline void L1DummyProducer::SimpleDigi(int,
                                        CLHEP::HepRandomEngine* engine,
                                        std::unique_ptr<L1MuRegionalCandCollection>& data,
                                        int type) const {
  if (verbose())
    std::cout << "L1DummyProducer::SimpleDigi<L1MuRegionalCandCollection>....\n" << std::flush;
  //typedef std::vector<L1MuRegionalCand>     L1MuRegionalCandCollection;
  assert(type >= 0 && type < 4);
  unsigned type_idx = type;  //tType: 0 DT, 1 bRPC, 2 CSC, 3 fRPC
  int bx = 0;
  unsigned phi, eta, pt, charge, ch_valid, finehalo, quality;
  float phiv(0.), etav(0.), ptv(0.);  //linear translation? 0.2pi,-2.5..2.5,0..100
  for (int i = 0; i < 4; i++) {
    phi = (int)(144 * engine->flat());  //8bits, 0..143
    eta = (int)(63 * engine->flat());   //6bits code
    phiv = phi * 2 * TMath::Pi() / 144.;
    etav = 2.5 * (-1 + 2 * eta / 63.);
    pt = ((int)(32 * engine->flat())) & 0x1f;  //5bits: 0..31
    ptv = 100 * (pt / 31.);
    charge = (engine->flat() > 0.5 ? 0 : 1);
    ;
    ch_valid = 0;
    finehalo = 0;
    quality = (int)(8 * engine->flat());  //3bits: 0..7
    L1MuRegionalCand cand(type_idx, phi, eta, pt, charge, ch_valid, finehalo, quality, bx);
    cand.setPhiValue(phiv);
    cand.setEtaValue(etav);
    cand.setPtValue(ptv);
    data->push_back(cand);
  }
  //L1MuRegionalCand(unsigned type_idx, unsigned phi, unsigned eta, unsigned pt,
  //unsigned charge, unsigned ch_valid, unsigned finehalo, unsigned quality, int bx);
  if (verbose())
    std::cout << "L1DummyProducer::SimpleDigi<L1MuRegionalCandCollection> end.\n" << std::flush;
}

template <>
inline void L1DummyProducer::SimpleDigi(int nevt,
                                        CLHEP::HepRandomEngine* engine,
                                        std::unique_ptr<L1MuDTTrackContainer>& data,
                                        int type) const {
  assert(type == 0);
  int type_idx = type;  //choose data type: 0 DT, 1 bRPC, 2 CSC, 3 fRPC
  if (verbose())
    std::cout << "L1DummyProducer::SimpleDigi<L1MuDTTrackContainer>....\n" << std::flush;
  std::unique_ptr<L1MuRegionalCandCollection> tracks(new L1MuRegionalCandCollection());
  SimpleDigi(nevt, engine, tracks, type_idx);
  typedef std::vector<L1MuDTTrackCand> L1MuDTTrackCandCollection;
  std::unique_ptr<L1MuDTTrackCandCollection> tracksd(new L1MuDTTrackCandCollection());
  for (L1MuRegionalCandCollection::const_iterator it = tracks->begin(); it != tracks->end(); it++) {
    L1MuDTTrackCand* cnd = new L1MuDTTrackCand();
    cnd->setDataWord(it->getDataWord());
    cnd->setBx(it->bx());
    tracksd->push_back(L1MuDTTrackCand());
    tracksd->push_back(*cnd);
  }
  data->setContainer(*tracksd);
  if (verbose())
    std::cout << "L1DummyProducer::SimpleDigi<L1MuDTTrackContainer> end.\n" << std::flush;
  //L1MuDTTrackCand( unsigned dataword, int bx, int uwh, int usc, int utag,
  //                 int adr1, int adr2, int adr3, int adr4, int utc );
}

template <>
inline void L1DummyProducer::SimpleDigi(int,
                                        CLHEP::HepRandomEngine* engine,
                                        std::unique_ptr<L1MuDTChambPhContainer>& data,
                                        int type) const {
  if (verbose())
    std::cout << "L1DummyProducer::SimpleDigi<L1MuDTChambPhContainer>....\n" << std::flush;
  typedef std::vector<L1MuDTChambPhDigi> Phi_Container;
  int ntrk = 4;
  Phi_Container tracks(ntrk);
  int ubx, uwh, usc, ust, uphr, uphb, uqua, utag, ucnt;
  for (int i = 0; i < ntrk; i++) {
    ubx = 0;   //bxNum()  - bx
    uwh = 0;   //whNum()  - wheel
    usc = 0;   //scNum()  - sector
    ust = 0;   //stNum()  - station
    uphr = 0;  //phi()    - radialAngle
    uphb = 0;  //phiB()   - bendingAngle
    uqua = 0;  //code()   - qualityCode
    utag = 0;  //Ts2Tag() - Ts2TagCode
    ucnt = 0;  //BxCnt()  - BxCntCode
    uwh = (int)(-2 + 5 * engine->flat());
    usc = (int)(12 * engine->flat());
    ust = (int)(1. + 4 * engine->flat());
    uqua = (int)(8 * engine->flat());
    L1MuDTChambPhDigi cand(ubx, uwh, usc, ust, uphr, uphb, uqua, utag, ucnt);
    tracks.push_back(cand);
  }
  data->setContainer(tracks);
  //L1MuDTChambPhDigi( int ubx, int uwh, int usc, int ust,
  //    int uphr, int uphb, int uqua, int utag, int ucnt );
  if (verbose())
    std::cout << "L1DummyProducer::SimpleDigi<L1MuDTChambPhContainer> end.\n" << std::flush;
}

template <>
inline void L1DummyProducer::SimpleDigi(int,
                                        CLHEP::HepRandomEngine* engine,
                                        std::unique_ptr<L1MuDTChambThContainer>& data,
                                        int type) const {
  if (verbose())
    std::cout << "L1DummyProducer::SimpleDigi<L1MuDTChambThContainer>....\n" << std::flush;
  typedef std::vector<L1MuDTChambThDigi> The_Container;
  int ntrk = 4;
  The_Container tracks(ntrk);
  int ubx, uwh, usc, ust, uos[7], uqa[7];
  for (int i = 0; i < ntrk; i++) {
    ubx = 0;
    uwh = (int)(-2 + 5 * engine->flat());
    usc = (int)(12 * engine->flat());
    ust = (int)(1. + 4 * engine->flat());
    for (int j = 0; j < 7; j++) {
      uos[j] = (engine->flat() > 0.5 ? 0 : 1);
      uqa[j] = (engine->flat() > 0.5 ? 0 : 1);
    }
    L1MuDTChambThDigi cand(ubx, uwh, usc, ust, uos, uqa);
    tracks.push_back(cand);
  }
  data->setContainer(tracks);
  //L1MuDTChambThDigi( int ubx, int uwh, int usc, int ust,
  //             int* uos, [int* uqual] );
  //"DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
  if (verbose())
    std::cout << "L1DummyProducer::SimpleDigi<L1MuDTChambThContainer> end.\n" << std::flush;
}

template <>
inline void L1DummyProducer::SimpleDigi(int nevt,
                                        CLHEP::HepRandomEngine* engine,
                                        std::unique_ptr<L1MuGMTCandCollection>& data,
                                        int type) const {
  if (verbose())
    std::cout << "L1DummyProducer::SimpleDigi<L1MuGMTCandCollection>....\n" << std::flush;
  //typedef std::vector<L1MuGMTCand>          L1MuGMTCandCollection;
  L1MuGMTCand cand(0, nevt);
  //cand.setPhiPacked();//8bits
  //cand.setPtPacked ();//5bits
  //cand.setQuality  ();//3bits
  //cand.setEtaPacked();//6bits
  //cand.setIsolation();//1bit
  //cand.setMIP      ();//1bit
  //cand.setChargePacked();//0:+, 1:-, 2:undef, 3:sync
  //cand.setBx       (nevt);
  //set physical values
  double eng = EBase_ + ESigm_ * CLHEP::RandGaussQ::shoot(engine);
  double phi = 2 * TMath::Pi() * engine->flat();
  double eta = 2.5 * (-1 + 2 * engine->flat());
  cand.setPtValue(eng);
  cand.setPhiValue(phi);
  cand.setEtaValue(eta);
  unsigned engp = (unsigned)(EBase_ + ESigm_ * CLHEP::RandGaussQ::shoot(engine));
  unsigned phip = (unsigned)(255 * engine->flat());
  unsigned etap = (unsigned)(63 * engine->flat());
  cand.setPtPacked(engp & 0x1f);
  cand.setPhiPacked(phip & 0x7f);
  cand.setEtaPacked(etap & 0x3f);
  double r = engine->flat();
  cand.setIsolation(r > 0.2);
  cand.setMIP(r > 0.7);
  cand.setChargePacked(r > 0.5 ? 0 : 1);
  cand.setBx(0);
  data->push_back(cand);
  if (verbose())
    std::cout << "L1DummyProducer::SimpleDigi<L1MuGMTCandCollection> end.\n" << std::flush;
}

template <>
inline void L1DummyProducer::SimpleDigi(int nevt,
                                        CLHEP::HepRandomEngine* engine,
                                        std::unique_ptr<L1MuGMTReadoutCollection>& data,
                                        int type) const {
  if (verbose())
    std::cout << "L1DummyProducer::SimpleDigi<L1MuGMTReadoutCollection>....\n" << std::flush;
  L1MuGMTReadoutRecord rec(0);
  int bxn = nevt;
  rec.setBxNr(bxn);
  rec.setEvNr(bxn);
  rec.setBxInEvent(0);
  std::unique_ptr<L1MuRegionalCandCollection> trks_dttf(new L1MuRegionalCandCollection);
  std::unique_ptr<L1MuRegionalCandCollection> trks_rpcb(new L1MuRegionalCandCollection);
  std::unique_ptr<L1MuRegionalCandCollection> trks_csc(new L1MuRegionalCandCollection);
  std::unique_ptr<L1MuRegionalCandCollection> trks_rpcf(new L1MuRegionalCandCollection);
  SimpleDigi(nevt, engine, trks_dttf, 0);
  SimpleDigi(nevt, engine, trks_rpcb, 1);
  SimpleDigi(nevt, engine, trks_csc, 2);
  SimpleDigi(nevt, engine, trks_rpcf, 3);
  for (int i = 0; i < 4; i++) {
    rec.setInputCand(i, trks_dttf->at(i));       //dt  : 0..3
    rec.setInputCand(i + 4, trks_rpcb->at(i));   //rpcb: 4..7
    rec.setInputCand(i + 8, trks_csc->at(i));    //csc : 8..11
    rec.setInputCand(i + 12, trks_rpcf->at(i));  //rpcf:12..15
  }
  for (int nr = 0; nr < 4; nr++) {
    int eng = (int)(EBase_ + ESigm_ * CLHEP::RandGaussQ::shoot(engine));
    rec.setGMTBrlCand(nr, eng & 0x11, eng & 0x11);  //set GMT barrel candidate
    rec.setGMTFwdCand(nr, eng & 0x11, eng & 0x11);  //set GMT forward candidate
    rec.setGMTCand(nr, eng & 0x11);                 //set GMT candidate (does not store rank)
    int eta = (int)(14 * engine->flat());           //0..13
    int phi = (int)(18 * engine->flat());           //0..17
    rec.setMIPbit(eta, phi);
    rec.setQuietbit(eta, phi);
  }
  data->addRecord(rec);
  ///tbd: add GMT extended cand(!)
  //rec.setBCERR(int bcerr);
  //rec.setGMTBrlCand(int nr, L1MuGMTExtendedCand const& cand);
  //rec.setGMTFwdCand(int nr, L1MuGMTExtendedCand const& cand);
  //rec.setGMTCand   (int nr, L1MuGMTExtendedCand const& cand);
  //rec.setInputCand (int nr, L1MuRegionalCand const& cand);
  //L1MuGMTReadoutCollection :: std::vector<L1MuGMTReadoutRecord> m_Records;
  //L1MuGMTReadoutCollection(int nbx) { m_Records.reserve(nbx); };
  //L1MuGMTExtendedCand(unsigned data, unsigned rank, int bx=0) : L1MuGMTCand (data, bx), m_rank(rank) {}
  if (verbose())
    std::cout << "L1DummyProducer::SimpleDigi<L1MuGMTReadoutCollection> end.\n" << std::flush;
}

template <>
inline void L1DummyProducer::SimpleDigi(int,
                                        CLHEP::HepRandomEngine*,
                                        std::unique_ptr<LTCDigiCollection>& data,
                                        int type) const {
  if (verbose())
    std::cout << "L1DummyProducer::SimpleDigi<LTCDigiCollection>....\n" << std::flush;
  //LTCs are FED id 816-823
  /*  
      6 64-bit words
      uint64_t *ld = (uint64_t*)data;
      
      word0: 59:56  4 bit    ld[0]>>56 & 0xf         trigType
             55:32 24 bit    ld[0]>>32 & 0x00ffffff  eventID
             31:20 12 bit    ld[0]>>20 & 0xfff       bunchNumber
             19: 8 12 bit    ld[0]>> 8 & 0x00000fff  sourceID (816-823?)

      word1: 63:32 32 bit    ld[1]>>32 & 0xffffffff  orbitNumber
             31:24 8 bit     ld[1]>>24 & 0xff        versionNumber
              3: 0 4 bit     ld[1      & 0xf         daqPartition  

      word2: 63:32 32 bit    ld[0]>>32 & 0xffffffff  runNumber
             31: 0 32 bit    ld[0]     & 0xffffffff  eventNumber

      word3: 63:32 32 bit    ld[3]>>32 & 0xffffffff  trigInhibitNumber
             31: 0 32 bit    ld[3]     & 0xffffffff  trigInputStat  

      word4: 63:0 64 bit     ld[4]                   bstGpsTime

      word5: (empty)
  */
  //need to make up something meaningfull to produce here..
  //LTCDigi(const unsigned char* data);
  if (verbose())
    std::cout << "L1DummyProducer::SimpleDigi<LTCDigiCollection> end.\n" << std::flush;
}

template <>
inline void L1DummyProducer::SimpleDigi(int,
                                        CLHEP::HepRandomEngine* engine,
                                        std::unique_ptr<CSCCorrelatedLCTDigiCollection>& data,
                                        int type) const {
  if (verbose())
    std::cout << "L1DummyProducer::SimpleDigi<CSCCorrelatedLCTDigiCollection>....\n" << std::flush;
  //typedef MuonDigiCollection<CSCDetId,CSCCorrelatedLCTDigi> CSCCorrelatedLCTDigiCollection;
  //CSCCorrelatedLCTDigi(const int trknmb, const int valid, const int quality, const int keywire, const int strip, const int clct_pattern, const int bend, const int bx, const int& mpclink = 0, const uint16_t & bx0=0, const uint16_t & syncErr = 0, const uint16_t & cscID=0);
  CSCCorrelatedLCTDigi dg = CSCCorrelatedLCTDigi();
  //tbd: set non-trivial random values
  dg.clear();  // set contents to zero
  //CSCDetId( int iendcap, int istation, int iring, int ichamber, int ilayer = 0 );
  enum eMinNum { MIN_ENDCAP = 1, MIN_STATION = 1, MIN_RING = 1, MIN_CHAMBER = 1, MIN_LAYER = 1 };
  enum eMaxNum { MAX_ENDCAP = 2, MAX_STATION = 4, MAX_RING = 4, MAX_CHAMBER = 36, MAX_LAYER = 6 };
  float rnd = engine->flat();
  int ec = (int)(MIN_ENDCAP + (MAX_ENDCAP - MIN_ENDCAP) * rnd + 1);
  int st = (int)(MIN_STATION + (MAX_STATION - MIN_STATION) * rnd + 1);
  int rg = (int)(MIN_RING + (MAX_RING - MIN_RING) * rnd + 1);
  int ch = (int)(MIN_CHAMBER + (MAX_CHAMBER - MIN_CHAMBER) * rnd + 1);
  int lr = (int)(MIN_LAYER + (MAX_LAYER - MIN_LAYER) * rnd + 1);
  CSCDetId did = CSCDetId(ec, st, rg, ch, lr);
  //CSCDetId did = CSCDetId();   //DetId(DetId::Muon, MuonSubdetId::CSC)
  //MuonDigiCollection::insertDigi(const IndexType& index, const DigiType& digi)
  data->insertDigi(did, dg);
  if (verbose())
    std::cout << "L1DummyProducer::SimpleDigi<CSCCorrelatedLCTDigiCollection> end.\n" << std::flush;
}

template <>
inline void L1DummyProducer::SimpleDigi(int nevt,
                                        CLHEP::HepRandomEngine* engine,
                                        std::unique_ptr<L1CSCTrackCollection>& data,
                                        int type) const {
  if (verbose())
    std::cout << "L1DummyProducer::SimpleDigi<L1CSCTrackCollection>...\n" << std::flush;
  std::unique_ptr<CSCCorrelatedLCTDigiCollection> dgcoll(new CSCCorrelatedLCTDigiCollection);
  SimpleDigi(nevt, engine, dgcoll, 0);
  csc::L1Track l1trk = csc::L1Track();
  std::unique_ptr<L1MuRegionalCandCollection> regcoll(new L1MuRegionalCandCollection);
  SimpleDigi(nevt, engine, regcoll, 2);
  L1MuRegionalCand regcand = *(regcoll->begin());
  l1trk.setDataWord(regcand.getDataWord());
  l1trk.setBx(regcand.bx());
  l1trk.setPhiValue(regcand.phiValue());
  l1trk.setEtaValue(regcand.etaValue());
  l1trk.setPtValue(regcand.ptValue());
  L1CSCTrack l1csctrk = std::make_pair(l1trk, *dgcoll);
  data->push_back(l1csctrk);
  //typedef std::vector<L1CSCTrack> L1CSCTrackCollection;
  //typedef std::pair<csc::L1Track,CSCCorrelatedLCTDigiCollection> L1CSCTrack;
  //L1Track() : L1MuRegionalCand(), m_name("csc::L1Track") { setType(2); setPtPacked(0); }
  //L1MuRegionalCand(unsigned dataword = 0, int bx = 0);
  if (verbose())
    std::cout << "L1DummyProducer::SimpleDigi<L1CSCTrackCollection> end.\n" << std::flush;
}

#endif
