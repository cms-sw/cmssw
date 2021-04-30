#include "DataFormats/L1TCorrelator/interface/TkElectron.h"
#include "DataFormats/L1TCorrelator/interface/TkElectronFwd.h"
#include "DataFormats/L1TCorrelator/interface/TkEm.h"
#include "DataFormats/L1TCorrelator/interface/TkEmFwd.h"
#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1TParticleFlow/interface/PFCandidate.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/transform.h"

#include "DataFormats/L1TParticleFlow/interface/layer1_emulator.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/egamma/l2egsorter_ref.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/egamma/l2egencoder_ref.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/egamma/L1EGPuppiIsoAlgo.h"

#include "L1Trigger/DemonstratorTools/interface/BoardDataWriter.h"
#include "L1Trigger/DemonstratorTools/interface/utilities.h"

#include <iostream>
#include <vector>

using namespace l1ct;

class L1TCtL2EgProducer : public edm::global::EDProducer<> {
public:
  explicit L1TCtL2EgProducer(const edm::ParameterSet &);
  ~L1TCtL2EgProducer() override;

private:
  ap_uint<64> encodeLayer1(const EGIsoObjEmu &egiso) const;
  ap_uint<128> encodeLayer1(const EGIsoEleObjEmu &egiso) const;

  std::vector<ap_uint<64>> encodeLayer1(const std::vector<EGIsoObjEmu> &photons) const;
  std::vector<ap_uint<64>> encodeLayer1(const std::vector<EGIsoEleObjEmu> &electrons) const;

  std::vector<ap_uint<64>> encodeLayer1EgObjs(unsigned int nObj,
                                              const std::vector<EGIsoObjEmu> &photons,
                                              const std::vector<EGIsoEleObjEmu> &electrons) const;

  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;

  void endJob() override;

  struct RefRemapper {
    typedef TTTrack<Ref_Phase2TrackerDigi_> L1TTTrackType;

    BXVector<edm::Ref<BXVector<l1t::EGamma>>> oldRefs;
    std::map<edm::Ref<BXVector<l1t::EGamma>>, edm::Ref<BXVector<l1t::EGamma>>> old2newRefMap;
    std::vector<std::pair<edm::Ref<l1t::EGammaBxCollection>, edm::Ptr<L1TTTrackType>>> origRefAndPtr;
  };

  void convertToEmu(const l1t::TkElectron &tkele, RefRemapper &refRemapper, l1ct::OutputBoard &boarOut) const;
  void convertToEmu(const l1t::TkEm &tkele, RefRemapper &refRemapper, l1ct::OutputBoard &boarOut) const;
  void convertToPuppi(const l1t::PFCandidateCollection &l1PFCands, l1ct::PuppiObjs &puppiObjs) const;

  template <class T>
  class PFInstanceInputs {
  public:
    typedef std::vector<std::pair<edm::EDGetTokenT<T>, std::vector<int>>> InputTokenAndChannels;
    PFInstanceInputs(L1TCtL2EgProducer *prod, const std::vector<edm::ParameterSet> &confs) {
      for (const auto &conf : confs) {
        const auto &producer_tag = conf.getParameter<edm::InputTag>("pfProducer");
        tokensAndChannels_.push_back(std::make_pair(
            prod->consumes<T>(edm::InputTag(producer_tag.label(), producer_tag.instance(), producer_tag.process())),
            conf.getParameter<std::vector<int>>("channels")));
      }
    }

    const InputTokenAndChannels &tokensAndChannels() const { return tokensAndChannels_; }

  private:
    InputTokenAndChannels tokensAndChannels_;
  };

  class PatternWriter {
  public:
    PatternWriter(const edm::ParameterSet &conf) : dataWriter_(nullptr) {
      unsigned int nFramesPerBX = conf.getParameter<uint32_t>("nFramesPerBX");

      std::map<l1t::demo::LinkId, std::pair<l1t::demo::ChannelSpec, std::vector<size_t>>> channelSpecs;

      for (const auto &channelConf : conf.getParameter<std::vector<edm::ParameterSet>>("channels")) {
        unsigned int inTMUX = channelConf.getParameter<uint32_t>("TMUX");
        unsigned int eventGap =
            inTMUX * nFramesPerBX - channelConf.getParameter<uint32_t>("nWords");  // assuming 96bit (= 3/2 word)
                                                                                   // words  = TMUX*9-2*3/2*words
        std::vector<uint32_t> chns = channelConf.getParameter<std::vector<uint32_t>>("channels");
        channelSpecs[l1t::demo::LinkId{channelConf.getParameter<std::string>("interface"),
                                       channelConf.getParameter<uint32_t>("id")}] =
            std::make_pair(l1t::demo::ChannelSpec{inTMUX, eventGap},
                           std::vector<size_t>(std::begin(chns), std::end(chns)));
      }

      dataWriter_ = std::make_unique<l1t::demo::BoardDataWriter>(
          l1t::demo::parseFileFormat(conf.getParameter<std::string>("format")),
          conf.getParameter<std::string>("outputFilename"),
          nFramesPerBX,
          conf.getParameter<uint32_t>("TMUX"),
          conf.getParameter<uint32_t>("maxLinesPerFile"),
          channelSpecs);
    }

    void addEvent(const l1t::demo::EventData &eventData) { dataWriter_->addEvent(eventData); }

    void flush() { dataWriter_->flush(); }

  private:
    std::unique_ptr<l1t::demo::BoardDataWriter> dataWriter_;
  };

  template <class TT, class T>
  void merge(const PFInstanceInputs<T> &instance,
             edm::Event &iEvent,
             RefRemapper &refRemapper,
             std::unique_ptr<TT> &out) const {
    edm::Handle<T> handle;
    for (const auto &tokenAndChannel : instance.tokensAndChannels()) {
      iEvent.getByToken(tokenAndChannel.first, handle);
      populate(out, handle, tokenAndChannel.second, refRemapper);
    }
    remapRefs(iEvent, out, refRemapper);
  }

  template <class TT>
  void remapRefs(edm::Event &iEvent, std::unique_ptr<TT> &out, RefRemapper &refRemapper) const {}

  void remapRefs(edm::Event &iEvent, std::unique_ptr<BXVector<l1t::EGamma>> &out, RefRemapper &refRemapper) const {
    edm::RefProd<BXVector<l1t::EGamma>> ref_egs = iEvent.getRefBeforePut<BXVector<l1t::EGamma>>(tkEGInstanceLabel_);
    edm::Ref<BXVector<l1t::EGamma>>::key_type idx = 0;
    for (std::size_t ix = 0; ix < out->size(); ix++) {
      refRemapper.old2newRefMap[refRemapper.oldRefs[ix]] = edm::Ref<BXVector<l1t::EGamma>>(ref_egs, idx++);
    }
  }

  template <class TT, class T>
  void populate(std::unique_ptr<T> &out,
                const edm::Handle<TT> &in,
                const std::vector<int> &links,
                RefRemapper &refRemapper) const {
    assert(links.size() == in->nRegions());
    for (unsigned int iBoard = 0, nBoard = in->nRegions(); iBoard < nBoard; ++iBoard) {
      auto region = in->region(iBoard);
      int linkID = links[iBoard];
      if (linkID < 0)
        continue;
      // std::cout << "Board eta: " << in->eta(iBoard) << " phi: " << in->phi(iBoard) << " link: " << linkID << std::endl;
      for (const auto &obj : region) {
        convertToEmu(obj, refRemapper, out->at(linkID));
      }
    }
  }

  void populate(std::unique_ptr<BXVector<l1t::EGamma>> &out,
                const edm::Handle<BXVector<l1t::EGamma>> &in,
                const std::vector<int> &links,
                RefRemapper &refRemapper) const {
    edm::Ref<BXVector<l1t::EGamma>>::key_type idx = 0;
    for (int bx = in->getFirstBX(); bx <= in->getLastBX(); bx++) {
      for (auto egee_itr = in->begin(bx); egee_itr != in->end(bx); egee_itr++) {
        out->push_back(bx, *egee_itr);
        // this to ensure that the old ref and the new object have the same
        // index in the BXVector collection so that we can still match them no
        // matter which BX we will insert next
        refRemapper.oldRefs.push_back(bx, edm::Ref<BXVector<l1t::EGamma>>(in, idx++));
      }
    }
  }

  template <class Tout, class Tin>
  void putEgObjects(edm::Event &iEvent,
                    const RefRemapper &refRemapper,
                    const std::string &label,
                    const std::vector<Tin> emulated) const {
    auto egobjs = std::make_unique<Tout>();
    for (const auto &emu : emulated) {
      if (emu.hwPt == 0)
        continue;
      auto obj = convertFromEmu(emu, refRemapper);
      egobjs->push_back(obj);
    }
    iEvent.put(std::move(egobjs), label);
  }

  l1t::TkEm convertFromEmu(const l1ct::EGIsoObjEmu &emu, const RefRemapper &refRemapper) const;
  l1t::TkElectron convertFromEmu(const l1ct::EGIsoEleObjEmu &emu, const RefRemapper &refRemapper) const;

  PFInstanceInputs<BXVector<l1t::EGamma>> tkEGInputs_;
  PFInstanceInputs<l1t::TkEmRegionalOutput> tkEmInputs_;
  PFInstanceInputs<l1t::TkElectronRegionalOutput> tkEleInputs_;
  std::string tkEGInstanceLabel_;
  std::string tkEmInstanceLabel_;
  std::string tkEleInstanceLabel_;
  l1ct::L2EgSorterEmulator l2egsorter;
  l1ct::L2EgEncoderEmulator l2encoder;
  edm::EDGetTokenT<std::vector<l1t::PFCandidate>> pfObjsToken_;
  l1ct::L1EGPuppiIsoAlgo l2EgPuppiIsoAlgo_;
  l1ct::L1EGPuppiIsoAlgo l2ElePuppiIsoAlgo_;
  bool doInPtrn_;
  bool doOutPtrn_;
  std::unique_ptr<PatternWriter> inPtrnWrt_;
  std::unique_ptr<PatternWriter> outPtrnWrt_;
};

L1TCtL2EgProducer::L1TCtL2EgProducer(const edm::ParameterSet &conf)
    : tkEGInputs_(this, conf.getParameter<std::vector<edm::ParameterSet>>("tkEgs")),
      tkEmInputs_(this, conf.getParameter<std::vector<edm::ParameterSet>>("tkEms")),
      tkEleInputs_(this, conf.getParameter<std::vector<edm::ParameterSet>>("tkElectrons")),
      tkEGInstanceLabel_(conf.getParameter<std::string>("egStaInstanceLabel")),
      tkEmInstanceLabel_(conf.getParameter<std::string>("tkEmInstanceLabel")),
      tkEleInstanceLabel_(conf.getParameter<std::string>("tkEleInstanceLabel")),
      l2egsorter(conf.getParameter<edm::ParameterSet>("sorter")),
      l2encoder(conf.getParameter<edm::ParameterSet>("encoder")),
      pfObjsToken_(consumes<std::vector<l1t::PFCandidate>>(conf.getParameter<edm::InputTag>("l1PFObjects"))),
      l2EgPuppiIsoAlgo_(conf.getParameter<edm::ParameterSet>("puppiIsoParametersTkEm")),
      l2ElePuppiIsoAlgo_(conf.getParameter<edm::ParameterSet>("puppiIsoParametersTkEle")),
      doInPtrn_(conf.getParameter<bool>("writeInPattern")),
      doOutPtrn_(conf.getParameter<bool>("writeOutPattern")),
      inPtrnWrt_(nullptr),
      outPtrnWrt_(nullptr) {
  produces<BXVector<l1t::EGamma>>(tkEGInstanceLabel_);
  produces<l1t::TkEmCollection>(tkEmInstanceLabel_);
  produces<l1t::TkElectronCollection>(tkEleInstanceLabel_);

  if (doInPtrn_) {
    inPtrnWrt_ = std::make_unique<PatternWriter>(conf.getParameter<edm::ParameterSet>("inPatternFile"));
  }
  if (doOutPtrn_) {
    outPtrnWrt_ = std::make_unique<PatternWriter>(conf.getParameter<edm::ParameterSet>("outPatternFile"));
  }
}

L1TCtL2EgProducer::~L1TCtL2EgProducer() {}

ap_uint<64> L1TCtL2EgProducer::encodeLayer1(const EGIsoObjEmu &egiso) const {
  ap_uint<64> ret = 0;
  ret(EGIsoObjEmu::BITWIDTH, 0) = egiso.pack();
  return ret;
}

ap_uint<128> L1TCtL2EgProducer::encodeLayer1(const EGIsoEleObjEmu &egiso) const {
  ap_uint<128> ret = 0;
  ret(EGIsoEleObjEmu::BITWIDTH, 0) = egiso.pack();
  return ret;
}

std::vector<ap_uint<64>> L1TCtL2EgProducer::encodeLayer1(const std::vector<EGIsoObjEmu> &photons) const {
  std::vector<ap_uint<64>> ret;
  ret.reserve(photons.size());
  for (const auto &phot : photons) {
    ret.push_back(encodeLayer1(phot));
  }
  return ret;
}

std::vector<ap_uint<64>> L1TCtL2EgProducer::encodeLayer1(const std::vector<EGIsoEleObjEmu> &electrons) const {
  std::vector<ap_uint<64>> ret;
  ret.reserve(2 * electrons.size());
  for (const auto &ele : electrons) {
    auto eleword = encodeLayer1(ele);
    ret.push_back(eleword(63, 0));
    ret.push_back(eleword(127, 64));
  }
  return ret;
}

std::vector<ap_uint<64>> L1TCtL2EgProducer::encodeLayer1EgObjs(unsigned int nObj,
                                                               const std::vector<EGIsoObjEmu> &photons,
                                                               const std::vector<EGIsoEleObjEmu> &electrons) const {
  std::vector<ap_uint<64>> ret;
  auto encoded_photons = encodeLayer1(photons);
  encoded_photons.resize(nObj, {0});
  auto encoded_eles = encodeLayer1(electrons);
  encoded_eles.resize(2 * nObj, {0});

  std::copy(encoded_photons.begin(), encoded_photons.end(), std::back_inserter(ret));
  std::copy(encoded_eles.begin(), encoded_eles.end(), std::back_inserter(ret));

  return ret;
}

void L1TCtL2EgProducer::produce(edm::StreamID, edm::Event &iEvent, const edm::EventSetup &) const {
  RefRemapper refmapper;

  auto outEgs = std::make_unique<BXVector<l1t::EGamma>>();
  merge(tkEGInputs_, iEvent, refmapper, outEgs);
  iEvent.put(std::move(outEgs), tkEGInstanceLabel_);

  auto boards = std::make_unique<std::vector<l1ct::OutputBoard>>(l2egsorter.nInputBoards());

  merge(tkEleInputs_, iEvent, refmapper, boards);
  merge(tkEmInputs_, iEvent, refmapper, boards);

  if (doInPtrn_) {
    l1t::demo::EventData inData;
    for (unsigned int ibrd = 0; ibrd < boards->size(); ibrd++) {
      inData.add(
          {"eglayer1", ibrd},
          encodeLayer1EgObjs(l2egsorter.nInputObjPerBoard(), (*boards)[ibrd].egphoton, (*boards)[ibrd].egelectron));
    }
    inPtrnWrt_->addEvent(inData);
  }

  std::vector<EGIsoObjEmu> out_photons_emu;
  std::vector<EGIsoEleObjEmu> out_eles_emu;
  l2egsorter.run(*boards, out_photons_emu, out_eles_emu);

  // PUPPI isolation
  auto &pfObjs = iEvent.get(pfObjsToken_);
  l1ct::PuppiObjs puppiObjs;
  convertToPuppi(pfObjs, puppiObjs);
  l2EgPuppiIsoAlgo_.run(out_photons_emu, puppiObjs);
  l2ElePuppiIsoAlgo_.run(out_eles_emu, puppiObjs);

  if (doOutPtrn_) {
    l1t::demo::EventData outData;
    outData.add({"eglayer2", 0}, l2encoder.encodeLayer2EgObjs(out_photons_emu, out_eles_emu));
    outPtrnWrt_->addEvent(outData);
  }

  putEgObjects<l1t::TkEmCollection>(iEvent, refmapper, tkEmInstanceLabel_, out_photons_emu);
  putEgObjects<l1t::TkElectronCollection>(iEvent, refmapper, tkEleInstanceLabel_, out_eles_emu);
}

void L1TCtL2EgProducer::endJob() {
  // Writing pending events to file before exiting
  if (doOutPtrn_)
    outPtrnWrt_->flush();
  if (doInPtrn_)
    inPtrnWrt_->flush();
}

void L1TCtL2EgProducer::convertToEmu(const l1t::TkElectron &tkele,
                                     RefRemapper &refRemapper,
                                     l1ct::OutputBoard &boarOut) const {
  EGIsoEleObjEmu emu;
  emu.initFromBits(tkele.egBinaryWord<EGIsoEleObj::BITWIDTH>());
  emu.srcCluster = nullptr;
  emu.srcTrack = nullptr;
  auto refEg = tkele.EGRef();
  const auto newref = refRemapper.old2newRefMap.find(refEg);
  if (newref != refRemapper.old2newRefMap.end()) {
    refEg = newref->second;
  }
  refRemapper.origRefAndPtr.push_back(std::make_pair(refEg, tkele.trkPtr()));
  emu.sta_idx = refRemapper.origRefAndPtr.size() - 1;
  // NOTE: The emulator and FW data-format stores absolute iso while the CMSSW object stores relative iso
  emu.setHwIso(EGIsoEleObjEmu::IsoType::TkIso, l1ct::Scales::makeIso(tkele.trkIsol() * tkele.pt()));
  emu.setHwIso(EGIsoEleObjEmu::IsoType::PfIso, l1ct::Scales::makeIso(tkele.pfIsol() * tkele.pt()));
  emu.setHwIso(EGIsoEleObjEmu::IsoType::PuppiIso, l1ct::Scales::makeIso(tkele.puppiIsol() * tkele.pt()));
  // std::cout << "[convertToEmu] TkEle pt: " << emu.hwPt << " eta: " << emu.hwEta << " phi: " << emu.hwPhi << " staidx: " << emu.sta_idx << std::endl;

  boarOut.egelectron.push_back(emu);
}

void L1TCtL2EgProducer::convertToEmu(const l1t::TkEm &tkem,
                                     RefRemapper &refRemapper,
                                     l1ct::OutputBoard &boarOut) const {
  EGIsoObjEmu emu;
  emu.initFromBits(tkem.egBinaryWord<EGIsoObj::BITWIDTH>());
  emu.srcCluster = nullptr;
  auto refEg = tkem.EGRef();
  const auto newref = refRemapper.old2newRefMap.find(refEg);
  if (newref != refRemapper.old2newRefMap.end()) {
    refEg = newref->second;
  }
  refRemapper.origRefAndPtr.push_back(std::make_pair(refEg, edm::Ptr<RefRemapper::L1TTTrackType>(nullptr, 0)));
  emu.sta_idx = refRemapper.origRefAndPtr.size() - 1;
  // NOTE: The emulator and FW data-format stores absolute iso while the CMSSW object stores relative iso
  emu.setHwIso(EGIsoObjEmu::IsoType::TkIso, l1ct::Scales::makeIso(tkem.trkIsol() * tkem.pt()));
  emu.setHwIso(EGIsoObjEmu::IsoType::PfIso, l1ct::Scales::makeIso(tkem.pfIsol() * tkem.pt()));
  emu.setHwIso(EGIsoObjEmu::IsoType::PuppiIso, l1ct::Scales::makeIso(tkem.puppiIsol() * tkem.pt()));
  emu.setHwIso(EGIsoObjEmu::IsoType::TkIsoPV, l1ct::Scales::makeIso(tkem.trkIsolPV() * tkem.pt()));
  emu.setHwIso(EGIsoObjEmu::IsoType::PfIsoPV, l1ct::Scales::makeIso(tkem.pfIsolPV() * tkem.pt()));
  // std::cout << "[convertToEmu] TkEM pt: " << emu.hwPt << " eta: " << emu.hwEta << " phi: " << emu.hwPhi << " staidx: " << emu.sta_idx << std::endl;
  boarOut.egphoton.push_back(emu);
}

void L1TCtL2EgProducer::convertToPuppi(const l1t::PFCandidateCollection &l1PFCands, l1ct::PuppiObjs &puppiObjs) const {
  for (const auto &l1PFCand : l1PFCands) {
    l1ct::PuppiObj obj;
    obj.initFromBits(l1PFCand.encodedPuppi64());
    puppiObjs.emplace_back(obj);
  }
}

l1t::TkEm L1TCtL2EgProducer::convertFromEmu(const l1ct::EGIsoObjEmu &egiso, const RefRemapper &refRemapper) const {
  // std::cout << "[convertFromEmu] TkEm pt: " << egiso.hwPt << " eta: " << egiso.hwEta << " phi: " << egiso.hwPhi << " staidx: " << egiso.sta_idx << std::endl;
  // NOTE: the TkEM object is created with the accuracy as in GT object (not the Correlator internal one)!
  const auto gteg = egiso.toGT();
  reco::Candidate::PolarLorentzVector mom(
      l1gt::Scales::floatPt(gteg.v3.pt), l1gt::Scales::floatEta(gteg.v3.eta), l1gt::Scales::floatPhi(gteg.v3.phi), 0.);
  // NOTE: The emulator and FW data-format stores absolute iso while the CMSSW object stores relative iso
  l1t::TkEm tkem(reco::Candidate::LorentzVector(mom),
                 refRemapper.origRefAndPtr[egiso.sta_idx].first,
                 egiso.floatRelIso(l1ct::EGIsoObjEmu::IsoType::TkIso),
                 egiso.floatRelIso(l1ct::EGIsoObjEmu::IsoType::TkIsoPV));
  tkem.setHwQual(gteg.quality);
  tkem.setPFIsol(egiso.floatRelIso(l1ct::EGIsoObjEmu::IsoType::PfIso));
  tkem.setPFIsolPV(egiso.floatRelIso(l1ct::EGIsoObjEmu::IsoType::PfIsoPV));
  tkem.setPuppiIsol(egiso.floatRelIso(l1ct::EGIsoObjEmu::IsoType::PuppiIso));
  tkem.setEgBinaryWord(gteg.pack());
  return tkem;
}

l1t::TkElectron L1TCtL2EgProducer::convertFromEmu(const l1ct::EGIsoEleObjEmu &egele,
                                                  const RefRemapper &refRemapper) const {
  // std::cout << "[convertFromEmu] TkEle pt: " << egele.hwPt << " eta: " << egele.hwEta << " phi: " << egele.hwPhi << " staidx: " << egele.sta_idx << std::endl;
  // NOTE: the TkElectron object is created with the accuracy as in GT object (not the Correlator internal one)!
  const auto gteg = egele.toGT();
  reco::Candidate::PolarLorentzVector mom(
      l1gt::Scales::floatPt(gteg.v3.pt), l1gt::Scales::floatEta(gteg.v3.eta), l1gt::Scales::floatPhi(gteg.v3.phi), 0.);
  // NOTE: The emulator and FW data-format stores absolute iso while the CMSSW object stores relative iso
  l1t::TkElectron tkele(reco::Candidate::LorentzVector(mom),
                        refRemapper.origRefAndPtr[egele.sta_idx].first,
                        refRemapper.origRefAndPtr[egele.sta_idx].second,
                        egele.floatRelIso(l1ct::EGIsoEleObjEmu::IsoType::TkIso));
  tkele.setHwQual(gteg.quality);
  tkele.setPFIsol(egele.floatRelIso(l1ct::EGIsoEleObjEmu::IsoType::PfIso));
  tkele.setPuppiIsol(egele.floatRelIso(l1ct::EGIsoEleObjEmu::IsoType::PuppiIso));
  tkele.setEgBinaryWord(gteg.pack());
  return tkele;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1TCtL2EgProducer);
