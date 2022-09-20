#ifndef REF_PFTKEGSORTER_BARREL_REF_H
#define REF_PFTKEGSORTER_BARREL_REF_H

#include <cstdio>
#include <vector>

#include "DataFormats/L1TParticleFlow/interface/layer1_emulator.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/common/bitonic_hybrid_sort_ref.h"

#ifdef CMSSW_GIT_HASH
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#endif

namespace edm {
  class ParameterSet;
}

namespace l1ct {
  class PFTkEGSorterBarrelEmulator {
  public:
    PFTkEGSorterBarrelEmulator(const unsigned int nObjToSort = 10, const unsigned int nObjSorted = 16)
        : nObjToSort_(nObjToSort), nObjSorted_(nObjSorted), debug_(false) {}

    PFTkEGSorterBarrelEmulator(const edm::ParameterSet& iConfig);

    virtual ~PFTkEGSorterBarrelEmulator() {}

    void setDebug(bool debug = true) { debug_ = debug; };

    void toFirmware_pho(const OutputRegion& outregions, EGIsoObj (&photons_in)[nObjSorted_]) const {
      for (unsigned int i = 0; i < nObjToSort_; i++) {
        EGIsoObj pho;
        if (i < outregions.egphoton.size()) {
          pho = outregions.egphoton[i];
        } else
          pho.clear();

        photons_in[i] = pho;
      }
    }

    void toFirmware_ele(const OutputRegion& outregions, EGIsoEleObj (&eles_in)[nObjSorted_]) const {
      for (unsigned int i = 0; i < nObjToSort_; i++) {
        EGIsoEleObj ele;
        if (i < outregions.egelectron.size()) {
          ele = outregions.egelectron[i];
        } else
          ele.clear();

        eles_in[i] = ele;
      }
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
        global_objects.back().hwEta = region.hwGlbEta(reg_obj.hwEta);  //<=========== uncomment this
        global_objects.back().hwPhi = region.hwGlbPhi(reg_obj.hwPhi);  //<=========== uncomment this
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
      if (in_objs.size() == 1) {  //size is 1, fine!
        std::copy(in_objs[0].begin(), in_objs[0].end(), std::back_inserter(out));
        if (out.size() > nObjSorted_)
          out.resize(nObjSorted_);                                //size 16
      } else if (in_objs.size() == 2) {                           //size is 2, fine!
        merge_regions(in_objs[0], in_objs[1], out, nObjSorted_);  //10, 10, 16
      } else {
        std::vector<T> pair_merge_01;                                       //size is >2, merge 0 and 1 regions always
        merge_regions(in_objs[0], in_objs[1], pair_merge_01, nObjSorted_);  //10, 10, 16

        std::vector<std::vector<T>> to_merge;
        if (in_objs.size() == 3)
          to_merge.push_back(pair_merge_01);  //push 01 only if size is 3 //and then in_objs[id] will be pushed into it

        std::vector<T> pair_merge_tmp = pair_merge_01;  //
        for (unsigned int id = 2, idn = 3; id < in_objs.size(); id += 2, idn = id + 1) {
          if (idn >= in_objs.size()) {        //if size is odd number starting from 3
            to_merge.push_back(in_objs[id]);  //size 10
          } else {
            std::vector<T> pair_merge;
            merge_regions(in_objs[id],
                          in_objs[idn],
                          pair_merge,
                          nObjSorted_);  //10, 10, 16 // merge two regions: 23, 45, 67, and so on

            //pair_merge_tmp.resize(nObjToSort_);
            //pair_merge.resize(nObjToSort_);
            merge_regions(
                pair_merge_tmp,
                pair_merge,
                pair_merge_tmp,
                nObjSorted_);  //16, 16, 16 //merge 23 with 01 for first time // then merge 45, and then 67, and then 89, and so on
            to_merge.push_back(
                pair_merge_tmp);  //push back 0123, 012345, 01234567, and so on (remember 01 is the 0th element)
          }
        }
        if (in_objs.size() % 2 == 1) {
          //to_merge[to_merge.size()-2].resize(nObjToSort_);
          merge_regions(
              to_merge[to_merge.size() - 2],
              to_merge[to_merge.size() - 1],
              out,
              nObjSorted_);  //16, 16, 16 //e.g. is size is 3, we merge 01 and 2, if size is 7, we merge 012345 and 6
        } else
          out =
              pair_merge_tmp;  //if size is even number, e.g. 6, out is 012345 (or we can say to_merge[to_merge.size()-1])
      }
    }

    unsigned int nObjToSort_;
    unsigned int nObjSorted_;
    bool debug_;
  };
}  // namespace l1ct

#endif
