#ifndef L2EGENCODER_REF_H
#define L2EGENCODER_REF_H

#include "DataFormats/L1TParticleFlow/interface/layer1_emulator.h"
#include "DataFormats/L1TParticleFlow/interface/egamma.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/dbgPrintf.h"

namespace edm {
  class ParameterSet;
}

namespace l1ct {

  struct L2EgEncoderEmulator {
    L2EgEncoderEmulator(unsigned int nTKELE_OUT, unsigned int nTKPHO_OUT)
        : nTkEleOut_(nTKELE_OUT), nTkPhoOut_(nTKPHO_OUT), nEncodedWords_(nTKELE_OUT * 1.5 + nTKPHO_OUT * 1.5) {
      assert(nTkEleOut_ % 2 == 0);
      assert(nTkPhoOut_ % 2 == 0);
    };

    L2EgEncoderEmulator(const edm::ParameterSet& iConfig);

    void toFirmware(const std::vector<ap_uint<64>>& encoded_in, ap_uint<64> encoded_fw[]) const;

    std::vector<ap_uint<64>> encodeLayer2EgObjs(const std::vector<EGIsoObjEmu>& photons,
                                                const std::vector<EGIsoEleObjEmu>& electrons) const {
      std::vector<ap_uint<64>> ret;

      auto encoded_photons = encodeLayer2(photons);
      encoded_photons.resize(nTkPhoOut_, {0});
      auto encoded_eles = encodeLayer2(electrons);
      encoded_eles.resize(nTkEleOut_, {0});
      //
      encodeLayer2To64bits(encoded_eles, ret);
      encodeLayer2To64bits(encoded_photons, ret);
      return ret;
    }

    template <class T>
    std::vector<ap_uint<64>> encodeLayer2EgObjs_trivial(const std::vector<T>& egs, int n) const {
      std::vector<ap_uint<64>> ret;

      auto encoded_egs = encodeLayer2_trivial<T>(egs);
      encoded_egs.resize(n, {0});
      //
      encodeLayer2To64bits(encoded_egs, ret);

      return ret;
    }

  private:
    template <class T>
    ap_uint<96> encodeLayer2(const T& egiso) const {
      return egiso.toGT().pack();
    }

    template <class T>
    std::vector<ap_uint<96>> encodeLayer2(const std::vector<T>& egisos) const {
      std::vector<ap_uint<96>> ret;
      ret.reserve(egisos.size());
      for (const auto& egiso : egisos) {
        ret.push_back(encodeLayer2(egiso));
      }
      return ret;
    }
    //
    template <class T>
    ap_uint<96> encodeLayer2_trivial(const T& egiso) const {
      ap_uint<96> ret = 0;
      ret(T::BITWIDTH - 1, 0) = egiso.pack();
      return ret;
    }

    template <class T>
    std::vector<ap_uint<96>> encodeLayer2_trivial(const std::vector<T>& egisos) const {
      std::vector<ap_uint<96>> ret;
      for (const auto& egiso : egisos) {
        ret.push_back(encodeLayer2_trivial(egiso));
      }
      return ret;
    }

    void encodeLayer2To64bits(const std::vector<ap_uint<96>>& packed96, std::vector<ap_uint<64>>& packed64) const {
      for (unsigned int i = 0; i < packed96.size(); i += 2) {
        packed64.push_back(packed96[i](63, 0));
        packed64.push_back((ap_uint<32>(packed96[i + 1](95, 64)), ap_uint<32>(packed96[i](95, 64))));
        packed64.push_back(packed96[i + 1](63, 0));
      }
    }

    unsigned int nTkEleOut_;
    unsigned int nTkPhoOut_;
    unsigned int nEncodedWords_;
  };

}  // namespace l1ct
#endif
