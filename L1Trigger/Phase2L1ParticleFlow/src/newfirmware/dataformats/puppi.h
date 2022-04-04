#ifndef FIRMWARE_dataformats_puppi_h
#define FIRMWARE_dataformats_puppi_h

#include "datatypes.h"
#include "bit_encoding.h"
#include "layer1_objs.h"
#include "pf.h"

namespace l1ct {

  struct PuppiObj {
    pt_t hwPt;
    glbeta_t hwEta;  // wider range to support global coordinates
    glbphi_t hwPhi;
    ParticleID hwId;

    static const int BITS_Z0_START = 0;
    static const int BITS_DXY_START = BITS_Z0_START + z0_t::width;
    static const int BITS_TKQUAL_START = BITS_DXY_START + dxy_t::width;
    static const int DATA_CHARGED_BITS_TOTAL = BITS_TKQUAL_START + tkquality_t::width;

    static const int BITS_PUPPIW_START = 0;
    static const int BITS_EMID_START = BITS_PUPPIW_START + puppiWgt_t::width;
    static const int DATA_NEUTRAL_BITS_TOTAL = BITS_EMID_START + emid_t::width;

    static const int DATA_BITS_TOTAL =
        DATA_CHARGED_BITS_TOTAL >= DATA_NEUTRAL_BITS_TOTAL ? DATA_CHARGED_BITS_TOTAL : DATA_NEUTRAL_BITS_TOTAL;

    ap_uint<DATA_BITS_TOTAL> hwData;

    inline z0_t hwZ0() const {
#ifndef __SYNTHESIS__
      assert(hwId.charged() || hwPt == 0);
#endif
      return z0_t(hwData(BITS_Z0_START + z0_t::width - 1, BITS_Z0_START));
    }

    inline void setHwZ0(z0_t z0) {
#ifndef __SYNTHESIS__
      assert(hwId.charged() || hwPt == 0);
#endif
      hwData(BITS_Z0_START + z0_t::width - 1, BITS_Z0_START) = z0(z0_t::width - 1, 0);
    }

    inline dxy_t hwDxy() const {
#ifndef __SYNTHESIS__
      assert(hwId.charged() || hwPt == 0);
#endif
      return dxy_t(hwData(BITS_DXY_START + dxy_t::width - 1, BITS_DXY_START));
    }

    inline void setHwDxy(dxy_t dxy) {
#ifndef __SYNTHESIS__
      assert(hwId.charged() || hwPt == 0);
#endif
      hwData(BITS_DXY_START + dxy_t::width - 1, BITS_DXY_START) = dxy(7, 0);
    }

    inline tkquality_t hwTkQuality() const {
#ifndef __SYNTHESIS__
      assert(hwId.charged() || hwPt == 0);
#endif
      return tkquality_t(hwData(BITS_TKQUAL_START + tkquality_t::width - 1, BITS_TKQUAL_START));
    }

    inline void setHwTkQuality(tkquality_t qual) {
#ifndef __SYNTHESIS__
      assert(hwId.charged() || hwPt == 0);
#endif
      hwData(BITS_TKQUAL_START + tkquality_t::width - 1, BITS_TKQUAL_START) = qual(tkquality_t::width - 1, 0);
    }

    inline puppiWgt_t hwPuppiW() const {
#ifndef __SYNTHESIS__
      assert(hwId.neutral());
#endif
      return puppiWgt_t(hwData(BITS_PUPPIW_START + puppiWgt_t::width - 1, BITS_PUPPIW_START));
    }

    inline void setHwPuppiW(puppiWgt_t w) {
#ifndef __SYNTHESIS__
      assert(hwId.neutral());
#endif
      hwData(BITS_PUPPIW_START + puppiWgt_t::width - 1, BITS_PUPPIW_START) = w(puppiWgt_t::width - 1, 0);
    }

    inline puppiWgt_t hwEmID() const {
#ifndef __SYNTHESIS__
      assert(hwId.neutral());
#endif
      return puppiWgt_t(hwData(BITS_EMID_START + emid_t::width - 1, BITS_EMID_START));
    }

    inline void setHwEmID(emid_t w) {
#ifndef __SYNTHESIS__
      assert(hwId.neutral());
#endif
      hwData(BITS_EMID_START + emid_t::width - 1, BITS_EMID_START) = w(emid_t::width - 1, 0);
    }

    inline bool operator==(const PuppiObj &other) const {
      return hwPt == other.hwPt && hwEta == other.hwEta && hwPhi == other.hwPhi && hwId == other.hwId &&
             hwData == other.hwData;
    }

    inline bool operator>(const PuppiObj &other) const { return hwPt > other.hwPt; }
    inline bool operator<(const PuppiObj &other) const { return hwPt < other.hwPt; }

    inline void clear() {
      hwPt = 0;
      hwEta = 0;
      hwPhi = 0;
      hwId.clear();
      hwData = 0;
    }

    inline void fill(const PFRegion &region, const PFChargedObj &src) {
      hwEta = region.hwGlbEta(src.hwVtxEta());
      hwPhi = region.hwGlbPhi(src.hwVtxPhi());
      hwId = src.hwId;
      hwPt = src.hwPt;
      hwData = 0;
      setHwZ0(src.hwZ0);
      setHwDxy(src.hwDxy);
      setHwTkQuality(src.hwTkQuality);
    }
    inline void fill(const PFRegion &region, const PFNeutralObj &src, pt_t puppiPt, puppiWgt_t puppiWgt) {
      hwEta = region.hwGlbEta(src.hwEta);
      hwPhi = region.hwGlbPhi(src.hwPhi);
      hwId = src.hwId;
      hwPt = puppiPt;
      hwData = 0;
      setHwPuppiW(puppiWgt);
      setHwEmID(src.hwEmID);
    }
    inline void fill(const PFRegion &region, const HadCaloObj &src, pt_t puppiPt, puppiWgt_t puppiWgt) {
      hwEta = region.hwGlbEta(src.hwEta);
      hwPhi = region.hwGlbPhi(src.hwPhi);
      hwId = src.hwIsEM() ? ParticleID::PHOTON : ParticleID::HADZERO;
      hwPt = puppiPt;
      hwData = 0;
      setHwPuppiW(puppiWgt);
      setHwEmID(src.hwEmID);
    }

    int intPt() const { return Scales::intPt(hwPt); }
    int intEta() const { return hwEta.to_int(); }
    int intPhi() const { return hwPhi.to_int(); }
    float floatPt() const { return Scales::floatPt(hwPt); }
    float floatEta() const { return Scales::floatEta(hwEta); }
    float floatPhi() const { return Scales::floatPhi(hwPhi); }
    int intId() const { return hwId.rawId(); }
    int pdgId() const { return hwId.pdgId(); }
    int oldId() const { return hwPt > 0 ? hwId.oldId() : 0; }
    int intCharge() const { return hwId.intCharge(); }
    float floatZ0() const { return Scales::floatZ0(hwZ0()); }
    float floatDxy() const { return Scales::floatDxy(hwDxy()); }
    float floatPuppiW() const { return hwId.neutral() ? Scales::floatPuppiW(hwPuppiW()) : 1.0f; }

    static const int BITWIDTH = pt_t::width + glbeta_t::width + glbphi_t::width + 3 + DATA_BITS_TOTAL;
    inline ap_uint<BITWIDTH> pack() const {
      ap_uint<BITWIDTH> ret;
      unsigned int start = 0;
      _pack_into_bits(ret, start, hwPt);
      _pack_into_bits(ret, start, hwEta);
      _pack_into_bits(ret, start, hwPhi);
      _pack_into_bits(ret, start, hwId.bits);
      _pack_into_bits(ret, start, hwData);
      return ret;
    }
    inline void initFromBits(const ap_uint<BITWIDTH> &src) {
      unsigned int start = 0;
      _unpack_from_bits(src, start, hwPt);
      _unpack_from_bits(src, start, hwEta);
      _unpack_from_bits(src, start, hwPhi);
      _unpack_from_bits(src, start, hwId.bits);
      _unpack_from_bits(src, start, hwData);
    }
    inline static PuppiObj unpack(const ap_uint<BITWIDTH> &src) {
      PuppiObj ret;
      ret.initFromBits(src);
      return ret;
    }
  };
  inline void clear(PuppiObj &c) { c.clear(); }

}  // namespace l1ct

#endif
