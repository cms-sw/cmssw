#ifndef L1Trigger_Phase2L1ParticleFlow_egamma_pftkegsorter_barrel_ref_h
#define L1Trigger_Phase2L1ParticleFlow_egamma_pftkegsorter_barrel_ref_h

#include <cstdio>
#include <vector>

#include "pftkegsorter_ref.h"
#include "DataFormats/L1TParticleFlow/interface/layer1_emulator.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/common/bitonic_hybrid_sort_ref.h"

namespace l1ct {
  class PFTkEGSorterBarrelEmulator : public PFTkEGSorterEmulator {
  public:
    PFTkEGSorterBarrelEmulator(const unsigned int nObjToSort = 10, const unsigned int nObjSorted = 16)
        : PFTkEGSorterEmulator(nObjToSort, nObjSorted) {}

#ifdef CMSSW_GIT_HASH
    PFTkEGSorterBarrelEmulator(const edm::ParameterSet& iConfig)
        : PFTkEGSorterEmulator(iConfig.getParameter<uint32_t>("nObjToSort"),
                               iConfig.getParameter<uint32_t>("nObjSorted")) {}
#endif

    ~PFTkEGSorterBarrelEmulator() override {}

    void toFirmware_pho(const OutputRegion& outregions, EGIsoObj photons_in[/*nObjSorted_*/]) const {
      for (unsigned int i = 0; i < nObjToSort_; i++) {
        EGIsoObj pho;
        if (i < outregions.egphoton.size()) {
          pho = outregions.egphoton[i];
        } else
          pho.clear();

        photons_in[i] = pho;
      }
    }

    void toFirmware_ele(const OutputRegion& outregions, EGIsoEleObj eles_in[/*nObjSorted_*/]) const {
      for (unsigned int i = 0; i < nObjToSort_; i++) {
        EGIsoEleObj ele;
        if (i < outregions.egelectron.size()) {
          ele = outregions.egelectron[i];
        } else
          ele.clear();

        eles_in[i] = ele;
      }
    }

    void runPho(const std::vector<l1ct::PFInputRegion>& pfregions,
                const std::vector<l1ct::OutputRegion>& outregions,
                const std::vector<unsigned int>& region_index,
                std::vector<l1ct::EGIsoObjEmu>& eg_sorted_inBoard) override {
      run<l1ct::EGIsoObjEmu>(pfregions, outregions, region_index, eg_sorted_inBoard);
    }
    void runEle(const std::vector<l1ct::PFInputRegion>& pfregions,
                const std::vector<l1ct::OutputRegion>& outregions,
                const std::vector<unsigned int>& region_index,
                std::vector<l1ct::EGIsoEleObjEmu>& eg_sorted_inBoard) override {
      run<l1ct::EGIsoEleObjEmu>(pfregions, outregions, region_index, eg_sorted_inBoard);
    }

    template <typename T>
    void run(const std::vector<PFInputRegion>& pfregions,
             const std::vector<OutputRegion>& outregions,
             const std::vector<unsigned int>& region_index,
             std::vector<T>& eg_sorted_inBoard) {
      // we copy to be able to resize them
      std::vector<std::vector<T>> objs_in;
      objs_in.reserve(nObjToSort_);
      for (unsigned int i : region_index) {
        std::vector<T> objs;
        extractEGObjEmu(pfregions[i].region, outregions[i], objs);
        if (debug_)
          dbgCout() << "objs size " << objs.size() << "\n";
        resize_input(objs);
        objs_in.push_back(objs);
        if (debug_)
          dbgCout() << "objs (re)size and total objs size " << objs.size() << " " << objs_in.size() << "\n";
      }

      merge(objs_in, eg_sorted_inBoard);

      if (debug_) {
        dbgCout() << "objs.size() size " << eg_sorted_inBoard.size() << "\n";
        for (const auto& out : eg_sorted_inBoard)
          dbgCout() << "kinematics of sorted objects " << out.hwPt << " " << out.hwEta << " " << out.hwPhi << "\n";
      }
    }

  private:
    void extractEGObjEmu(const PFRegionEmu& region,
                         const l1ct::OutputRegion& outregion,
                         std::vector<l1ct::EGIsoObjEmu>& eg) {
      extractEGObjEmu(region, outregion.egphoton, eg);
    }
    void extractEGObjEmu(const PFRegionEmu& region,
                         const l1ct::OutputRegion& outregion,
                         std::vector<l1ct::EGIsoEleObjEmu>& eg) {
      extractEGObjEmu(region, outregion.egelectron, eg);
    }
    template <typename T>
    void extractEGObjEmu(const PFRegionEmu& region,
                         const std::vector<T>& regional_objects,
                         std::vector<T>& global_objects) {
      for (const auto& reg_obj : regional_objects) {
        global_objects.emplace_back(reg_obj);
        global_objects.back().hwEta = region.hwGlbEta(reg_obj.hwEta);
        global_objects.back().hwPhi = region.hwGlbPhi(reg_obj.hwPhi);
      }
    }

    template <typename T>
    void resize_input(std::vector<T>& in) const {
      if (in.size() > nObjToSort_) {
        in.resize(nObjToSort_);
      } else if (in.size() < nObjToSort_) {
        for (unsigned int i = 0, diff = (nObjToSort_ - in.size()); i < diff; ++i) {
          in.push_back(T());
          in.back().clear();
        }
      }
    }

    template <typename T>
    void merge_regions(const std::vector<T>& in_region1,
                       const std::vector<T>& in_region2,
                       std::vector<T>& out,
                       unsigned int nOut) const {
      // we crate a bitonic list
      out = in_region1;
      if (debug_)
        for (const auto& tmp : out)
          dbgCout() << "out " << tmp.hwPt << " " << tmp.hwEta << " " << tmp.hwPhi << "\n";
      std::reverse(out.begin(), out.end());
      if (debug_)
        for (const auto& tmp : out)
          dbgCout() << "out reverse " << tmp.hwPt << " " << tmp.hwEta << " " << tmp.hwPhi << "\n";
      std::copy(in_region2.begin(), in_region2.end(), std::back_inserter(out));
      if (debug_)
        for (const auto& tmp : out)
          dbgCout() << "out inserted " << tmp.hwPt << " " << tmp.hwEta << " " << tmp.hwPhi << "\n";

      hybridBitonicMergeRef(&out[0], out.size(), 0, false);

      if (out.size() > nOut) {
        out.resize(nOut);
        if (debug_)
          for (const auto& tmp : out)
            dbgCout() << "final " << tmp.hwPt << " " << tmp.hwEta << " " << tmp.hwPhi << "\n";
      }
    }

    template <typename T>
    void merge(const std::vector<std::vector<T>>& in_objs, std::vector<T>& out) const {
      unsigned int nregions = in_objs.size();
      std::vector<T> pair_merge(nObjSorted_);
      if (nregions == 18) {  // merge pairs, and accumulate pairs
        for (unsigned int i = 0; i < nregions; i += 2) {
          merge_regions(in_objs[i + 0], in_objs[i + 1], pair_merge, nObjSorted_);
          if (i == 0)
            out = pair_merge;
          else
            merge_regions(out, pair_merge, out, nObjSorted_);
        }
      } else if (nregions == 9) {  // simple accumulation
        for (unsigned int i = 0; i < nregions; ++i) {
          for (unsigned int j = 0, nj = in_objs[i].size(); j < nObjSorted_; ++j) {
            if (j < nj)
              pair_merge[j] = in_objs[i][j];
            else
              pair_merge[j].clear();
          }
          if (i == 0)
            out = pair_merge;
          else
            merge_regions(out, pair_merge, out, nObjSorted_);
        }
      } else
        throw std::runtime_error("This sorter requires 18 or 9 regions");
    }
  };
}  // namespace l1ct

#endif
