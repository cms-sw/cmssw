#ifndef FIRMWARE_PFTKEGSORTER_REF_H
#define FIRMWARE_PFTKEGSORTER_REF_H

#include <cstdio>
#include <vector>

#include "DataFormats/L1TParticleFlow/interface/layer1_emulator.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/dbgPrintf.h"

#ifdef CMSSW_GIT_HASH
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#endif

namespace edm {
  class ParameterSet;
}

namespace l1ct {
  class PFTkEGSorterEmulator {
  public:
    PFTkEGSorterEmulator(const unsigned int nObjToSort = 6, const unsigned int nObjSorted = 16)
        : nObjToSort_(nObjToSort), nObjSorted_(nObjSorted), debug_(false) {}

#ifdef CMSSW_GIT_HASH
    PFTkEGSorterEmulator(const edm::ParameterSet& iConfig)
        : PFTkEGSorterEmulator(iConfig.getParameter<uint32_t>("nObjToSort"),
                               iConfig.getParameter<uint32_t>("nObjSorted")) {}

#endif

    ~PFTkEGSorterEmulator(){};

    void setDebug(bool debug = true) { debug_ = debug; };

    template <typename T>
    void run(const std::vector<l1ct::PFInputRegion>& pfregions,
             const std::vector<l1ct::OutputRegion>& outregions,
             const std::vector<unsigned int>& region_index,
             std::vector<T>& eg_sorted_inBoard) {
      std::vector<T> eg_unsorted_inBoard = eg_sorted_inBoard;
      mergeEGObjFromRegions<T>(pfregions, outregions, region_index, eg_unsorted_inBoard);

      if (debug_ && !eg_unsorted_inBoard.empty()) {
        dbgCout() << "\nUNSORTED " << typeid(T).name() << "\n";
        for (int j = 0, nj = eg_unsorted_inBoard.size(); j < nj; j++)
          dbgCout() << "EG[" << j << "]: pt = " << eg_unsorted_inBoard[j].hwPt
                    << ",\t eta = " << eg_unsorted_inBoard[j].hwEta << ",\t phi = " << eg_unsorted_inBoard[j].hwPhi
                    << "\n";
      }

      if (debug_ && !eg_unsorted_inBoard.empty())
        dbgCout() << "\nSORTED " << typeid(T).name() << "\n";

      eg_sorted_inBoard = eg_unsorted_inBoard;
      std::reverse(eg_sorted_inBoard.begin(), eg_sorted_inBoard.end());
      std::stable_sort(eg_sorted_inBoard.begin(), eg_sorted_inBoard.end(), comparePt<T>);
      if (eg_sorted_inBoard.size() > nObjSorted_)
        eg_sorted_inBoard.resize(nObjSorted_);

      if (debug_ && !eg_unsorted_inBoard.empty()) {
        for (int j = 0, nj = eg_sorted_inBoard.size(); j < nj; j++)
          dbgCout() << "EG[" << j << "]: pt = " << eg_sorted_inBoard[j].hwPt
                    << ",\t eta = " << eg_sorted_inBoard[j].hwEta << ",\t phi = " << eg_sorted_inBoard[j].hwPhi << "\n";
      }
    }

  private:
    unsigned int nObjToSort_, nObjSorted_;
    bool debug_;

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
    static bool comparePt(T obj1, T obj2) {
      return (obj1.hwPt > obj2.hwPt);
    }

    template <typename T>
    void mergeEGObjFromRegions(const std::vector<l1ct::PFInputRegion>& pfregions,
                               const std::vector<l1ct::OutputRegion>& outregions,
                               const std::vector<unsigned int>& region_index,
                               std::vector<T>& eg_unsorted_inBoard) {
      for (unsigned int i : region_index) {
        const auto& region = pfregions[i].region;

        std::vector<T> eg_tmp;
        extractEGObjEmu(region, outregions[i], eg_tmp);
        if (debug_ && !eg_tmp.empty())
          dbgCout() << "\nOutput Region " << i << ": eta = " << region.floatEtaCenter()
                    << " and phi = " << region.floatPhiCenter() << " \n";

        for (int j = 0, nj = std::min<int>(eg_tmp.size(), nObjToSort_); j < nj; j++) {
          if (debug_)
            dbgCout() << "EG[" << j << "] pt = " << eg_tmp[j].hwPt << ",\t eta = " << eg_tmp[j].hwEta
                      << ",\t phi = " << eg_tmp[j].hwPhi << "\n";
          eg_unsorted_inBoard.push_back(eg_tmp[j]);
        }
        if (debug_ && !eg_tmp.empty())
          dbgCout() << "\n";
      }
    }
  };
}  // namespace l1ct

#endif
