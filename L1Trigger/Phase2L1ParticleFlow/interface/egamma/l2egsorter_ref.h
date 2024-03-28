#ifndef L2EgSorter_REF_H
#define L2EgSorter_REF_H

#include "DataFormats/L1TParticleFlow/interface/layer1_emulator.h"
#include "DataFormats/L1TParticleFlow/interface/egamma.h"
#include "DataFormats/L1TParticleFlow/interface/pf.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/common/bitonic_hybrid_sort_ref.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/dbgPrintf.h"

#include <algorithm>

namespace edm {
  class ParameterSet;
}

namespace l1ct {

  class L2EgSorterEmulator {
  public:
    L2EgSorterEmulator(unsigned int nRegions, unsigned int nEGPerRegion, unsigned int nEGOut, bool debug)
        : nREGIONS(nRegions), nEGPerRegion(nEGPerRegion), nEGOut(nEGOut), debug_(debug) {}

    L2EgSorterEmulator(const edm::ParameterSet &iConfig);

    virtual ~L2EgSorterEmulator() {}

    template <int NRegions, int NObjs>
    void toFirmware(const std::vector<l1ct::OutputBoard> &in,
                    EGIsoObj (&photons_in)[NRegions][NObjs],
                    EGIsoEleObj (&eles_in)[NRegions][NObjs]) const {
      for (unsigned int ib = 0; ib < NRegions; ib++) {
        const auto &region = in[ib];
        for (unsigned int io = 0; io < NObjs; io++) {
          EGIsoObj pho;
          EGIsoEleObj ele;
          if (io < region.egphoton.size())
            pho = region.egphoton[io];
          else
            pho.clear();
          if (io < region.egelectron.size())
            ele = region.egelectron[io];
          else
            ele.clear();

          photons_in[ib][io] = pho;
          eles_in[ib][io] = ele;
        }
      }
    }

    void toFirmware(const std::vector<EGIsoObjEmu> &out_photons,
                    const std::vector<EGIsoEleObjEmu> &out_eles,
                    EGIsoObj out_egphs[/*nObjOut*/],
                    EGIsoEleObj out_egeles[/*nObjOut*/]) const;

    void run(const std::vector<l1ct::OutputBoard> &in,
             std::vector<EGIsoObjEmu> &out_photons,
             std::vector<EGIsoEleObjEmu> &out_eles) const;

    void setDebug(int verbose) { debug_ = verbose; }

    unsigned int nInputRegions() const { return nREGIONS; }
    unsigned int nInputObjPerRegion() const { return nEGPerRegion; }
    unsigned int nOutputObj() const { return nEGOut; }

  private:
    template <typename T>
    void resize_input(std::vector<T> &in) const {
      if (in.size() > nEGPerRegion) {
        in.resize(nEGPerRegion);
      } else if (in.size() < nEGPerRegion) {
        for (unsigned int i = 0, diff = (nEGPerRegion - in.size()); i < diff; ++i) {
          in.push_back(T());
          in.back().clear();
        }
      }
    }

    template <typename T>
    static bool comparePt(const T &obj1, const T &obj2) {
      return (obj1.hwPt > obj2.hwPt);
    }

    template <typename T>
    void print_objects(const std::vector<T> &objs, const std::string &label) const {
      for (unsigned int i = 0; i < objs.size(); ++i) {
        dbgCout() << label << " [" << i << "] pt: " << objs[i].hwPt << " eta: " << objs[i].hwEta
                  << " phi: " << objs[i].hwPhi << " qual: " << objs[i].hwQual.to_string(2) << std::endl;
      }
    }

    template <typename T>
    void merge_regions(const std::vector<T> &in_region1,
                       const std::vector<T> &in_region2,
                       std::vector<T> &out,
                       unsigned int nOut) const {
      // we crate a bitonic list
      out = in_region1;
      std::reverse(out.begin(), out.end());
      std::copy(in_region2.begin(), in_region2.end(), std::back_inserter(out));
      hybridBitonicMergeRef(&out[0], out.size(), 0, false);

      if (out.size() > nOut)
        out.resize(nOut);
    }

    template <typename T>
    void merge(const std::vector<std::vector<T>> &in_objs, std::vector<T> &out) const {
      if (in_objs.size() == 1) {
        std::copy(in_objs[0].begin(), in_objs[0].end(), std::back_inserter(out));
        if (out.size() > nEGOut)
          out.resize(nEGOut);
      } else if (in_objs.size() == 2) {
        merge_regions(in_objs[0], in_objs[1], out, nEGOut);
      } else {
        std::vector<std::vector<T>> to_merge;
        for (unsigned int id = 0, idn = 1; id < in_objs.size(); id += 2, idn = id + 1) {
          if (idn >= in_objs.size()) {
            to_merge.push_back(in_objs[id]);
          } else {
            std::vector<T> pair_merge;
            merge_regions(in_objs[id], in_objs[idn], pair_merge, nEGPerRegion);
            to_merge.push_back(pair_merge);
          }
        }
        merge(to_merge, out);
      }
    }

    const unsigned int nREGIONS;
    const unsigned int nEGPerRegion;
    const unsigned int nEGOut;
    int debug_;
  };
}  // namespace l1ct

#endif
