#ifndef DataFormats_L1TParticleFlow_layer1_emulator_h
#define DataFormats_L1TParticleFlow_layer1_emulator_h

#include <fstream>
#include <vector>
#include "DataFormats/L1TParticleFlow/interface/layer1_objs.h"
#include "DataFormats/L1TParticleFlow/interface/pf.h"
#include "DataFormats/L1TParticleFlow/interface/puppi.h"
#include "DataFormats/L1TParticleFlow/interface/egamma.h"
#include "DataFormats/L1TParticleFlow/interface/emulator_io.h"

namespace l1t {
  class PFTrack;
  class PFCluster;
  class PFCandidate;
  class SAMuon;
}  // namespace l1t

namespace l1ct {

  struct HadCaloObjEmu : public HadCaloObj {
    const l1t::PFCluster *src;
    bool read(std::fstream &from);
    bool write(std::fstream &to) const;
    void clear() {
      HadCaloObj::clear();
      src = nullptr;
    }
  };

  struct EmCaloObjEmu : public EmCaloObj {
    const l1t::PFCluster *src;
    bool read(std::fstream &from);
    bool write(std::fstream &to) const;
    void clear() {
      EmCaloObj::clear();
      src = nullptr;
    }
  };

  struct TkObjEmu : public TkObj {
    uint16_t hwChi2, hwStubs;
    float simPt, simCaloEta, simCaloPhi, simVtxEta, simVtxPhi, simZ0, simD0;
    const l1t::PFTrack *src;
    bool read(std::fstream &from);
    bool write(std::fstream &to) const;
    void clear() {
      TkObj::clear();
      src = nullptr;
      hwChi2 = 0;
      hwStubs = 0;
      simPt = 0;
      simCaloEta = 0;
      simCaloPhi = 0;
      simVtxEta = 0;
      simVtxPhi = 0;
      simZ0 = 0;
      simD0 = 0;
    }
  };

  struct MuObjEmu : public MuObj {
    const l1t::SAMuon *src;
    bool read(std::fstream &from);
    bool write(std::fstream &to) const;
    void clear() {
      MuObj::clear();
      src = nullptr;
    }
  };

  struct PFChargedObjEmu : public PFChargedObj {
    const l1t::PFCluster *srcCluster;
    const l1t::PFTrack *srcTrack;
    const l1t::SAMuon *srcMu;
    const l1t::PFCandidate *srcCand;
    bool read(std::fstream &from);
    bool write(std::fstream &to) const;
    void clear() {
      PFChargedObj::clear();
      srcCluster = nullptr;
      srcTrack = nullptr;
      srcMu = nullptr;
      srcCand = nullptr;
    }
  };

  struct PFNeutralObjEmu : public PFNeutralObj {
    const l1t::PFCluster *srcCluster;
    const l1t::PFCandidate *srcCand;
    bool read(std::fstream &from);
    bool write(std::fstream &to) const;
    void clear() {
      PFNeutralObj::clear();
      srcCluster = nullptr;
      srcCand = nullptr;
    }
  };

  struct PFRegionEmu : public PFRegion {
    PFRegionEmu() : PFRegion() {}
    PFRegionEmu(float etaCenter, float phicenter);
    PFRegionEmu(float etamin, float etamax, float phicenter, float phiwidth, float etaextra, float phiextra);

    // global coordinates
    bool contains(float eta, float phi) const;
    bool containsHw(glbeta_t glbeta, glbphi_t phi) const;
    float localEta(float globalEta) const;
    float localPhi(float globalPhi) const;

    bool read(std::fstream &from);
    bool write(std::fstream &to) const;
  };

  struct PuppiObjEmu : public PuppiObj {
    const l1t::PFCluster *srcCluster;
    const l1t::PFTrack *srcTrack;
    const l1t::SAMuon *srcMu;
    const l1t::PFCandidate *srcCand;
    bool read(std::fstream &from);
    bool write(std::fstream &to) const;
    void clear() {
      PuppiObj::clear();
      srcCluster = nullptr;
      srcTrack = nullptr;
      srcMu = nullptr;
      srcCand = nullptr;
    }
    inline void fill(const PFRegionEmu &region, const PFChargedObjEmu &src) {
      PuppiObj::fill(region, src);
      srcCluster = src.srcCluster;
      srcTrack = src.srcTrack;
      srcMu = src.srcMu;
      srcCand = src.srcCand;
    }
    inline void fill(const PFRegionEmu &region, const PFNeutralObjEmu &src, pt_t puppiPt, puppiWgt_t puppiWgt) {
      PuppiObj::fill(region, src, puppiPt, puppiWgt);
      srcCluster = src.srcCluster;
      srcTrack = nullptr;
      srcMu = nullptr;
      srcCand = src.srcCand;
    }
    inline void fill(const PFRegionEmu &region, const HadCaloObjEmu &src, pt_t puppiPt, puppiWgt_t puppiWgt) {
      PuppiObj::fill(region, src, puppiPt, puppiWgt);
      srcCluster = src.src;
      srcTrack = nullptr;
      srcMu = nullptr;
      srcCand = nullptr;
    }
  };

  struct EGObjEmu : public EGIsoObj {
    const l1t::PFCluster *srcCluster;
    void clear() {
      srcCluster = nullptr;
      EGIsoObj::clear();
    }
  };

  struct EGIsoObjEmu : public EGIsoObj {
    const l1t::PFCluster *srcCluster;
    // we use an index to the standalone object needed to retrieve a Ref when putting
    int sta_idx;
    bool read(std::fstream &from);
    bool write(std::fstream &to) const;
    void clear() {
      EGIsoObj::clear();
      srcCluster = nullptr;
      sta_idx = -1;
      clearIsoVars();
    }

    void clearIsoVars() {
      hwIsoVars[0] = 0;
      hwIsoVars[1] = 0;
      hwIsoVars[2] = 0;
      hwIsoVars[3] = 0;
    }

    using EGIsoObj::floatIso;

    enum IsoType { TkIso = 0, PfIso = 1, TkIsoPV = 2, PfIsoPV = 3 };

    float floatIso(IsoType type) const { return Scales::floatIso(hwIsoVars[type]); }
    float floatRelIso(IsoType type) const { return Scales::floatIso(hwIsoVars[type]) / floatPt(); }
    float hwIsoVar(IsoType type) const { return hwIsoVars[type]; }
    void setHwIso(IsoType type, iso_t value) { hwIsoVars[type] = value; }

    iso_t hwIsoVars[4];
  };

  struct EGIsoEleObjEmu : public EGIsoEleObj {
    const l1t::PFCluster *srcCluster;
    const l1t::PFTrack *srcTrack;
    // we use an index to the standalone object needed to retrieve a Ref when putting
    int sta_idx;
    bool read(std::fstream &from);
    bool write(std::fstream &to) const;
    void clear() {
      EGIsoEleObj::clear();
      srcCluster = nullptr;
      srcTrack = nullptr;
      sta_idx = -1;
      clearIsoVars();
    }

    void clearIsoVars() {
      hwIsoVars[0] = 0;
      hwIsoVars[1] = 0;
    }

    using EGIsoEleObj::floatIso;

    enum IsoType { TkIso = 0, PfIso = 1 };

    float floatIso(IsoType type) const { return Scales::floatIso(hwIsoVars[type]); }
    float floatRelIso(IsoType type) const { return Scales::floatIso(hwIsoVars[type]) / floatPt(); }
    float hwIsoVar(IsoType type) const { return hwIsoVars[type]; }
    void setHwIso(IsoType type, iso_t value) { hwIsoVars[type] = value; }

    iso_t hwIsoVars[2];
  };

  struct PVObjEmu : public PVObj {
    bool read(std::fstream &from);
    bool write(std::fstream &to) const;
  };

  template <typename T>
  struct DetectorSector {
    PFRegionEmu region;
    std::vector<T> obj;
    DetectorSector() {}
    DetectorSector(float etamin, float etamax, float phicenter, float phiwidth, float etaextra = 0, float phiextra = 0)
        : region(etamin, etamax, phicenter, phiwidth, etaextra, phiextra) {}
    // convenience forwarding of some methods
    typedef typename std::vector<T>::const_iterator const_iterator;
    typedef typename std::vector<T>::iterator iterator;
    inline const T &operator[](unsigned int i) const { return obj[i]; }
    inline T &operator[](unsigned int i) { return obj[i]; }
    inline const_iterator begin() const { return obj.begin(); }
    inline iterator begin() { return obj.begin(); }
    inline const_iterator end() const { return obj.end(); }
    inline iterator end() { return obj.end(); }
    inline unsigned int size() const { return obj.size(); }
    inline void resize(unsigned int size) { obj.resize(size); }
    inline void clear() { obj.clear(); }
  };

  struct RawInputs {
    std::vector<DetectorSector<ap_uint<96>>> track;
    DetectorSector<ap_uint<64>> muon;  // muons are global
    std::vector<DetectorSector<ap_uint<256>>> hgcalcluster;

    bool read(std::fstream &from);
    bool write(std::fstream &to) const;
    void clear();
  };

  struct RegionizerDecodedInputs {
    std::vector<DetectorSector<HadCaloObjEmu>> hadcalo;
    std::vector<DetectorSector<EmCaloObjEmu>> emcalo;
    std::vector<DetectorSector<TkObjEmu>> track;
    DetectorSector<MuObjEmu> muon;  // muons are global

    bool read(std::fstream &from);
    bool write(std::fstream &to) const;
    void clear();
  };

  struct PFInputRegion {
    PFRegionEmu region;
    std::vector<HadCaloObjEmu> hadcalo;
    std::vector<EmCaloObjEmu> emcalo;
    std::vector<TkObjEmu> track;
    std::vector<MuObjEmu> muon;

    PFInputRegion() {}
    PFInputRegion(float etamin, float etamax, float phicenter, float phiwidth, float etaextra, float phiextra)
        : region(etamin, etamax, phicenter, phiwidth, etaextra, phiextra) {}
    bool read(std::fstream &from);
    bool write(std::fstream &to) const;
    void clear();
  };

  struct OutputRegion {
    std::vector<PFChargedObjEmu> pfcharged;
    std::vector<PFNeutralObjEmu> pfphoton;
    std::vector<PFNeutralObjEmu> pfneutral;
    std::vector<PFChargedObjEmu> pfmuon;
    std::vector<PuppiObjEmu> puppi;
    std::vector<EGObjEmu> egsta;
    std::vector<EGIsoObjEmu> egphoton;
    std::vector<EGIsoEleObjEmu> egelectron;

    bool read(std::fstream &from);
    bool write(std::fstream &to) const;
    void clear();

    // for multiplicities
    enum ObjType {
      anyType = 0,
      chargedType = 1,
      neutralType = 2,
      electronType = 3,
      muonType = 4,
      chargedHadronType = 5,
      neutralHadronType = 6,
      photonType = 7,
      nPFTypes = 8,
      egisoType = 8,
      egisoeleType = 9,
      nObjTypes = 10
    };
    static constexpr const char *objTypeName[nObjTypes] = {
        "", "Charged", "Neutral", "Electron", "Muon", "ChargedHadron", "NeutralHadron", "Photon", "EGIso", "EGIsoEle"};
    unsigned int nObj(ObjType type, bool puppi) const;
  };

  struct OutputBoard {
    float eta;
    float phi;
    // NOTE: region_index is not written to the dump file
    std::vector<unsigned int> region_index;
    std::vector<EGIsoObjEmu> egphoton;
    std::vector<EGIsoEleObjEmu> egelectron;

    bool read(std::fstream &from);
    bool write(std::fstream &to) const;
    void clear();
  };

  struct Event {
    enum { VERSION = 11 };
    uint32_t run, lumi;
    uint64_t event;
    RawInputs raw;
    RegionizerDecodedInputs decoded;
    std::vector<PFInputRegion> pfinputs;
    std::vector<PVObjEmu> pvs;
    std::vector<ap_uint<64>> pvs_emu;
    std::vector<OutputRegion> out;
    std::vector<OutputBoard> board_out;

    Event() : run(0), lumi(0), event(0) {}

    bool read(std::fstream &from);
    bool write(std::fstream &to) const;
    void clear();
    void init(uint32_t run, uint32_t lumi, uint64_t event);
    inline l1ct::PVObjEmu pv(unsigned int ipv = 0) const {
      l1ct::PVObjEmu ret;
      if (ipv < pvs.size())
        ret = pvs[ipv];
      else
        ret.clear();
      return ret;
    }
    inline ap_uint<64> pv_emu(unsigned int ipv = 0) const {
      ap_uint<64> ret = 0;
      if (ipv < pvs_emu.size())
        ret = pvs_emu[ipv];
      return ret;
    }
  };

  template <typename T1, typename T2>
  void toFirmware(const std::vector<T1> &in, unsigned int NMAX, T2 out[/*NMAX*/]) {
    unsigned int n = std::min<unsigned>(in.size(), NMAX);
    for (unsigned int i = 0; i < n; ++i)
      out[i] = in[i];
    for (unsigned int i = n; i < NMAX; ++i)
      out[i].clear();
  }

}  // namespace l1ct

#endif
