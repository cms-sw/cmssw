#ifndef L1_EMUL_BIAS_H
#define L1_EMUL_BIAS_H

/*\class L1EmulBias
 *\description produces modified emulator data to mimmic hw problems
 *\usage l1 monitoring software validation
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

// random generation
#include "CLHEP/Random/RandomEngine.h"
#include "CLHEP/Random/RandGaussQ.h"

class L1EmulBias : public edm::global::EDProducer<> {
public:
  explicit L1EmulBias(const edm::ParameterSet&);
  ~L1EmulBias() override;

protected:
  //virtual void beginRun(edm::Run&, const edm::EventSetup&);
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

public:
  template <class T>
  void ModifyCollection(std::unique_ptr<T>& data, const edm::Handle<T> emul, CLHEP::HepRandomEngine*) const;

private:
  int verbose_;
  int verbose() const { return verbose_; }
  edm::InputTag m_DEsource[dedefs::DEnsys][2];
  bool m_doSys[dedefs::DEnsys];
  std::string instName[dedefs::DEnsys][5];
};

/* Notes:
   .by default data is make identical to emul
   .biasing is to be implemented via specialization
   .bias may be defined for each data type, eg data word bit-shifting
   .keep template function specialization in header file
*/

template <class T>
void L1EmulBias::ModifyCollection(std::unique_ptr<T>& data, const edm::Handle<T> emul, CLHEP::HepRandomEngine*) const {
  data = (std::unique_ptr<T>)(const_cast<T*>(emul.product()));
}

template <>
inline void L1EmulBias::ModifyCollection(std::unique_ptr<EcalTrigPrimDigiCollection>& data,
                                         const edm::Handle<EcalTrigPrimDigiCollection> emul,
                                         CLHEP::HepRandomEngine*) const {
  typedef EcalTrigPrimDigiCollection::const_iterator col_cit;
  for (col_cit it = emul->begin(); it != emul->end(); it++) {
    EcalTriggerPrimitiveDigi col(*it);
    int iphi = it->id().iphi();
    bool reset = (iphi > 18 && iphi < 39);  //remove few supermodules
    for (int s = 0; s < 5; s++) {
      uint16_t sample = it->sample(s).raw();
      if (sample == 0)
        continue;
      uint16_t tmp = reset ? 0 : sample;
      if (reset)
        tmp = sample >> 1;
      col.setSampleValue(s, tmp);
      if (verbose() && sample != 0)
        std::cout << "[emulbias] etp " << *it << "\t sample: " << s << "  " << std::hex << sample << " -> "
                  << col.sample(s).raw() << std::dec << std::endl;
    }
    data->push_back(col);
  }
}

template <>
inline void L1EmulBias::ModifyCollection(std::unique_ptr<HcalTrigPrimDigiCollection>& data,
                                         const edm::Handle<HcalTrigPrimDigiCollection> emul,
                                         CLHEP::HepRandomEngine*) const {
  typedef HcalTrigPrimDigiCollection::const_iterator col_cit;
  for (col_cit it = emul->begin(); it != emul->end(); it++) {
    HcalTriggerPrimitiveDigi col(*it);
    int iphi = it->id().iphi();
    bool reset = (iphi > 18 && iphi < 27);  //remove few supermodules
    for (int s = 0; s < 5; s++) {
      uint16_t sample = it->sample(s).raw();
      if (sample == 0)
        continue;
      uint16_t tmp = reset ? 0 : sample;
      if (reset)
        tmp = sample >> 1;
      col.setSample(s, tmp);
    }
    data->push_back(col);
  }
}

template <>
inline void L1EmulBias::ModifyCollection(std::unique_ptr<L1CaloEmCollection>& data,
                                         const edm::Handle<L1CaloEmCollection> emul,
                                         CLHEP::HepRandomEngine* engine) const {
  typedef L1CaloEmCollection::const_iterator col_cit;
  for (col_cit it = emul->begin(); it != emul->end(); it++) {
    unsigned crate = it->rctCrate();
    unsigned raw = it->raw();
    bool iso = it->isolated();
    unsigned rdata = raw;
    if (crate < 4 * engine->flat())
      rdata = raw >> 1;
    L1CaloEmCand cand(rdata, crate, iso, it->index(), it->bx(), false);
    data->push_back(cand);
  }
  //L1CaloEmCand(uint16_t data, unsigned crate, bool iso);
  //L1CaloEmCand(uint16_t data, unsigned crate, bool iso, uint16_t index, int16_t bx, bool dummy);
}

template <>
inline void L1EmulBias::ModifyCollection(std::unique_ptr<L1CaloRegionCollection>& data,
                                         const edm::Handle<L1CaloRegionCollection> emul,
                                         CLHEP::HepRandomEngine* engine) const {
  typedef L1CaloRegionCollection::const_iterator col_cit;
  for (col_cit it = emul->begin(); it != emul->end(); it++) {
    unsigned crate = it->rctCrate();
    unsigned raw = it->et();
    uint16_t rdata = raw;
    if (crate < 4 * engine->flat())
      rdata = raw >> 1;
    L1CaloRegion cand(rdata, it->gctEta(), it->gctPhi(), it->bx());
    data->push_back(cand);
  }
  //L1CaloRegion(uint16_t data, unsigned ieta, unsigned iphi, int16_t bx);
  //Note: raw data accessor missing in dataformats!
}

template <>
inline void L1EmulBias::ModifyCollection(std::unique_ptr<L1GctEmCandCollection>& data,
                                         const edm::Handle<L1GctEmCandCollection> emul,
                                         CLHEP::HepRandomEngine* engine) const {
  typedef L1GctEmCandCollection::const_iterator col_cit;
  for (col_cit it = emul->begin(); it != emul->end(); it++) {
    unsigned raw = it->raw();
    uint16_t rdata = raw;
    if (it->phiIndex() < 4 * engine->flat())  //0-17
      rdata = raw >> 1;
    L1GctEmCand cand(rdata, it->isolated());
    data->push_back(cand);
  }
  //etaIndex(), etaSign() : -6 to -0, +0 to +6
  //L1GctEmCand(uint16_t data, bool iso);
}

template <>
inline void L1EmulBias::ModifyCollection(std::unique_ptr<L1GctJetCandCollection>& data,
                                         const edm::Handle<L1GctJetCandCollection> emul,
                                         CLHEP::HepRandomEngine* engine) const {
  typedef L1GctJetCandCollection::const_iterator col_cit;
  for (col_cit it = emul->begin(); it != emul->end(); it++) {
    unsigned raw = it->raw();
    uint16_t rdata = raw;
    if (it->phiIndex() < 4 * engine->flat())  //0-17
      rdata = raw >> 1;
    L1GctJetCand cand(rdata, it->isTau(), it->isForward());
    data->push_back(cand);
  }
  //L1GctJetCand(uint16_t data, bool isTau, bool isFor);
}

template <>
inline void L1EmulBias::ModifyCollection(std::unique_ptr<L1MuRegionalCandCollection>& data,
                                         const edm::Handle<L1MuRegionalCandCollection> emul,
                                         CLHEP::HepRandomEngine* engine) const {
  typedef L1MuRegionalCandCollection::const_iterator col_cit;
  for (col_cit it = emul->begin(); it != emul->end(); it++) {
    L1MuRegionalCand cand(*it);
    //unsigned raw = it->getDataWord();
    unsigned phi = it->phi_packed();
    if (phi > 90 && phi < 110)
      cand.setPtPacked((it->pt_packed()) >> 1);
    //raw = (raw>>2);
    //L1MuRegionalCand cand(raw);
    //cand.setType(it->type_idx());
    data->push_back(cand);
  }
  /* few alternatives...
  unsigned pt= it->pt_packed(); //0..31
  unsigned int qua = it->quality(); //0..7
  if(qua<4){cand.setPtPacked((pt>>2)&0x1f);cand.setQualityPacked((qua<<1)&0x07);}
  double rnd = CLHEP::RandGaussQ::shoot(engine);
  if(rnd>0.7) {
    raw_=(raw>>1);
    cand.setDataWord(raw_);
  } else if (rnd>0.3) {
    pt_ *= (int)(1+0.3*engine->flat());
    cand.setPtPacked(pt_);
  } else 
    cand.reset();
  unsigned raw = it->getDataWord();
  if(2.5<fabs(it->phiValue())<3.0)
    rdata = raw>>1;
  L1MuRegionalCand cand(rdata,it->bx());    
  */
  //L1MuRegionalCand(unsigned dataword = 0, int bx = 0);
  //L1MuRegionalCand(unsigned type_idx, unsigned phi, unsigned eta, unsigned pt, unsigned charge, unsigned ch_valid, unsigned finehalo, unsigned quality, int bx);
}

template <>
inline void L1EmulBias::ModifyCollection(std::unique_ptr<L1MuDTTrackContainer>& data,
                                         const edm::Handle<L1MuDTTrackContainer> emul,
                                         CLHEP::HepRandomEngine* engine) const {
  typedef std::vector<L1MuDTTrackCand> TrackContainer;
  typedef TrackContainer::const_iterator col_cit;
  TrackContainer const* tracks_in = emul->getContainer();
  TrackContainer tracks;
  for (col_cit it = tracks_in->begin(); it != tracks_in->end(); it++) {
    L1MuDTTrackCand cand(*it);
    cand.setType(it->type_idx());
    unsigned pt = it->pt_packed();  //0..31
    unsigned qua = it->quality();   //0..7
    if (qua < 4) {
      cand.setPtPacked((pt >> 2) & 0x1f);
      cand.setQualityPacked((qua << 1) & 0x07);
    }
    tracks.push_back(cand);
  }
  data->setContainer(tracks);
  /*   few alternatives...
  unsigned phip = it->phi_packed();
  unsigned raw = it->getDataWord();
  uint16_t rdata = raw;
  if(2.5<fabs(it->phiValue())<3.0)
    rdata = raw>>1;
  L1MuRegionalCand cand(rdata,it->bx());    
  double rnd    = engine->flat();
  */
}

template <>
inline void L1EmulBias::ModifyCollection(std::unique_ptr<L1MuDTChambPhContainer>& data,
                                         const edm::Handle<L1MuDTChambPhContainer> emul,
                                         CLHEP::HepRandomEngine* engine) const {
  typedef std::vector<L1MuDTChambPhDigi> Phi_Container;
  typedef Phi_Container::const_iterator col_it;
  Phi_Container const* tracks_in = emul->getContainer();
  Phi_Container tracks(tracks_in->size());
  int uqua;
  for (col_it it = tracks_in->begin(); it != tracks_in->end(); it++) {
    uqua = it->code();  // (int)(10*engine->flat());
    uqua = (uqua < 2 ? uqua + 1 : uqua);
    L1MuDTChambPhDigi cand(
        it->bxNum(), it->whNum(), it->scNum(), it->stNum(), it->phi(), it->phiB(), uqua, it->Ts2Tag(), it->BxCnt());
    tracks.push_back(cand);
  }
  data->setContainer(tracks);
}

template <>
inline void L1EmulBias::ModifyCollection(std::unique_ptr<L1MuDTChambThContainer>& data,
                                         const edm::Handle<L1MuDTChambThContainer> emul,
                                         CLHEP::HepRandomEngine*) const {
  typedef std::vector<L1MuDTChambThDigi> Thi_Container;
  typedef Thi_Container::const_iterator col_cit;
  Thi_Container const* tracks_in = emul->getContainer();
  Thi_Container tracks(tracks_in->size());
  int uos[7], uqa[7];
  for (col_cit it = tracks_in->begin(); it != tracks_in->end(); it++) {
    for (int j = 0; j < 7; j++) {
      uos[j] = (it->position(j) ? 0 : 1);
      uqa[j] = (it->quality(j) ? 0 : 1);
    }
    int stnum = it->stNum();
    stnum = (stnum > 2 ? stnum - 1 : stnum);
    L1MuDTChambThDigi cand(it->bxNum(), it->whNum(), it->scNum(), stnum, uos, uqa);
    tracks.push_back(cand);
  }
  data->setContainer(tracks);
}

template <>
inline void L1EmulBias::ModifyCollection(std::unique_ptr<LTCDigiCollection>& data,
                                         const edm::Handle<LTCDigiCollection> emul,
                                         CLHEP::HepRandomEngine*) const {
  typedef std::vector<LTCDigi>::const_iterator col_cit;
  for (col_cit it = emul->begin(); it != emul->end(); it++) {
    data->push_back(*it);
    //note: raw data accessor missing in dataformats!
    //data->push_back(LTCDigi(it->data()>>1));
  }
}

template <>
inline void L1EmulBias::ModifyCollection(std::unique_ptr<L1MuGMTCandCollection>& data,
                                         const edm::Handle<L1MuGMTCandCollection> emul,
                                         CLHEP::HepRandomEngine*) const {
  //typedef std::vector<L1MuGMTCand>          L1MuGMTCandCollection;
  typedef std::vector<L1MuGMTCand>::const_iterator col_cit;
  for (col_cit it = emul->begin(); it != emul->end(); it++) {
    float phiv = it->phiValue();
    unsigned dword = it->getDataWord();
    if (phiv > 2. && phiv < 4.)
      dword = dword >> 2;
    L1MuGMTCand cand(dword, it->bx());
    data->push_back(cand);
    //cand.setPtPacked(cand.ptIndex()>>1);
    //data->push_back(L1MuGMTCand((it->getDataWord()>>1),it->bx()));
  }
}

template <>
inline void L1EmulBias::ModifyCollection(std::unique_ptr<L1MuGMTReadoutCollection>& data,
                                         const edm::Handle<L1MuGMTReadoutCollection> emul,
                                         CLHEP::HepRandomEngine*) const {
  typedef std::vector<L1MuGMTReadoutRecord>::const_iterator col_cit;
  std::vector<L1MuGMTReadoutRecord> col = emul->getRecords();
  for (col_cit it = col.begin(); it != col.end(); it++) {
    L1MuGMTReadoutRecord rec(it->getBxInEvent());
    rec.setBxNr(it->getBxNr());
    rec.setEvNr(it->getEvNr());
    rec.setBCERR(it->getBCERR());

    std::unique_ptr<L1MuRegionalCandCollection> new_dttf(new L1MuRegionalCandCollection);
    std::unique_ptr<L1MuRegionalCandCollection> new_rpcb(new L1MuRegionalCandCollection);
    std::unique_ptr<L1MuRegionalCandCollection> new_csc(new L1MuRegionalCandCollection);
    std::unique_ptr<L1MuRegionalCandCollection> new_rpcf(new L1MuRegionalCandCollection);

    L1MuRegionalCandCollection old_dttf = it->getDTBXCands();
    L1MuRegionalCandCollection old_rpcb = it->getBrlRPCCands();
    L1MuRegionalCandCollection old_csc = it->getCSCCands();
    L1MuRegionalCandCollection old_rpcf = it->getFwdRPCCands();

    typedef L1MuRegionalCandCollection::const_iterator ait;
    for (ait it = old_dttf.begin(); it != old_dttf.end(); it++) {
      L1MuRegionalCand cand(*it);
      if (it->quality() < 4)
        cand.setPtPacked((it->pt_packed() >> 2) & 0x1f);
      cand.setType(it->type_idx());
      new_dttf->push_back(cand);
    }
    for (ait it = old_rpcb.begin(); it != old_rpcb.end(); it++) {
      L1MuRegionalCand cand(*it);
      if (it->quality() < 4)
        cand.setPtPacked((it->pt_packed() >> 2) & 0x1f);
      cand.setType(it->type_idx());
      new_rpcb->push_back(cand);
    }
    for (ait it = old_csc.begin(); it != old_csc.end(); it++) {
      L1MuRegionalCand cand(*it);
      if (it->quality() < 4)
        cand.setPtPacked((it->pt_packed() >> 2) & 0x1f);
      cand.setType(it->type_idx());
      new_csc->push_back(cand);
    }
    for (ait it = old_rpcf.begin(); it != old_rpcf.end(); it++) {
      L1MuRegionalCand cand(*it);
      if (it->quality() < 4)
        cand.setPtPacked((it->pt_packed() >> 2) & 0x1f);
      cand.setType(it->type_idx());
      new_rpcf->push_back(cand);
    }

    for (unsigned i = 0; i < old_dttf.size(); i++)
      rec.setInputCand(i, new_dttf->at(i));  //dt  : 0..3
    for (unsigned i = 0; i < old_rpcb.size(); i++)
      rec.setInputCand(i + 4, new_rpcb->at(i));  //rpcb: 4..7
    for (unsigned i = 0; i < old_csc.size(); i++)
      rec.setInputCand(i + 8, new_csc->at(i));  //csc : 8..11
    for (unsigned i = 0; i < old_rpcf.size(); i++)
      rec.setInputCand(i + 12, new_rpcf->at(i));  //rpcf:12..15

    data->addRecord(rec);
  }
  //void addRecord(L1MuGMTReadoutRecord const& rec) {
}

template <>
inline void L1EmulBias::ModifyCollection(std::unique_ptr<CSCCorrelatedLCTDigiCollection>& data,
                                         const edm::Handle<CSCCorrelatedLCTDigiCollection> emul,
                                         CLHEP::HepRandomEngine*) const {
  //typedef MuonDigiCollection<CSCDetId,CSCCorrelatedLCTDigi> CSCCorrelatedLCTDigiCollection;
  typedef CSCCorrelatedLCTDigiCollection::DigiRangeIterator mapIt;  //map iterator
  typedef CSCCorrelatedLCTDigiCollection::const_iterator vecIt;     //vec iterator
  //loop over data (map<idx,vec_digi>)
  for (mapIt mit = emul->begin(); mit != emul->end(); mit++) {
    //get detector index
    CSCDetId did = (*mit).first;
    //get vec_digi range(pair)  corresponding to idx of map
    //CSCCorrelatedLCTDigiCollection::Range ctpRange = emul->get(did)
    //loop over digi vector (ie between begin and end pointers in range)
    //for (vecIt vit = ctpRange.first; vit != ctpRange.second; vit++) {
    for (vecIt vit = emul->get((*mit).first).first; vit != emul->get((*mit).first).second; vit++) {
      ///modify digi
      CSCCorrelatedLCTDigi dg = *vit;
      //dg.clear;
      uint16_t tn = dg.getTrknmb();
      if (tn == 2)
        tn--;
      dg.setTrknmb(tn);
      //dg.setTrknmb   (dg.getTrknmb   ());
      //dg.setMPCLink  (dg.getMPCLink  ());
      //dg.setWireGroup(dg.getWireGroup());
      ///append digi
      data->insertDigi(did, dg);
    }
  }
}

template <>
inline void L1EmulBias::ModifyCollection(std::unique_ptr<L1CSCTrackCollection>& data,
                                         const edm::Handle<L1CSCTrackCollection> emul,
                                         CLHEP::HepRandomEngine*) const {
  typedef L1CSCTrackCollection::const_iterator col_cit;
  //typedef std::vector<L1CSCTrack> L1CSCTrackCollection;
  //typedef std::pair<csc::L1Track,CSCCorrelatedLCTDigiCollection> L1CSCTrack;
  //typedef MuonDigiCollection<CSCDetId,CSCCorrelatedLCTDigi> CSCCorrelatedLCTDigiCollection;
  typedef CSCCorrelatedLCTDigiCollection::DigiRangeIterator mapIt;  //map iterator
  typedef CSCCorrelatedLCTDigiCollection::const_iterator vecIt;     //vec iterator
  CSCCorrelatedLCTDigiCollection_ ctf_trk_data_v, ctf_trk_emul_v;   //vector
  //loop over csc-tracks (ie pairs<l1track,digi_vec>)
  for (col_cit tcit = emul->begin(); tcit != emul->end(); tcit++) {
    csc::L1Track l1trk = tcit->first;
    if (l1trk.quality() < 4)
      l1trk.setPtPacked((l1trk.pt_packed() >> 2) & 0x1f);
    l1trk.setType(l1trk.type_idx());
    //L1MuRegionalCand reg(tcit->first.getDataWord(), tcit->first.bx());
    std::unique_ptr<CSCCorrelatedLCTDigiCollection> dgcoll(new CSCCorrelatedLCTDigiCollection);
    CSCCorrelatedLCTDigiCollection ldc = tcit->second;  //muondigicollection=map
    //get the lct-digi-collection (ie muon-digi-collection)
    //loop over data (map<idx,vec_digi>)
    for (mapIt mit = ldc.begin(); mit != ldc.end(); mit++) {
      //get vec_digi range(pair)  corresponding to idx of map
      //loop over digi vector (ie between begin and end pointers in range)
      //CSCCorrelatedLCTDigiCollection::Range ctpRange = ctp_lct_data_->get((*mit).first)
      CSCDetId did = (*mit).first;
      //for (vecIt vit = ctpRange.first; vit != ctpRange.second; vit++) {
      for (vecIt vit = ldc.get((*mit).first).first; vit != ldc.get((*mit).first).second; vit++) {
        CSCCorrelatedLCTDigi dg = *vit;
        uint16_t tn = dg.getTrknmb();
        if (tn == 2)
          tn--;
        dg.setTrknmb(tn);
        dgcoll->insertDigi(did, dg);
      }
    }
    L1CSCTrack l1csctrk = std::make_pair(l1trk, *dgcoll);
    data->push_back(l1csctrk);
  }
}

#endif
