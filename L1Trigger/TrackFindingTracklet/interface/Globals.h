// Globals: holds "global" variables such as the IMATH_TrackletCalculators
#ifndef L1Trigger_TrackFindingTracklet_interface_Globals_h
#define L1Trigger_TrackFindingTracklet_interface_Globals_h

#include <memory>
#include <map>
#include <string>
#include <vector>
#include <array>
#include <fstream>
#include <unordered_map>

#ifdef USEHYBRID
#include "L1Trigger/TrackFindingTMTT/interface/KFParamsComb.h"
#include "L1Trigger/TrackFindingTMTT/interface/Settings.h"
#endif

namespace trklet {

  class TrackDerTable;
  class SLHCEvent;
  class HistBase;
  class Settings;
  class TrackletLUT;
  struct imathGlobals;
  class IMATH_TrackletCalculator;
  class IMATH_TrackletCalculatorDisk;
  class IMATH_TrackletCalculatorOverlap;

  class Globals {
  public:
    Globals(Settings const& settings);

    ~Globals();

    SLHCEvent*& event() { return theEvent_; }

    HistBase*& histograms() { return theHistBase_; }

    TrackDerTable*& trackDerTable() { return trackDerTable_; }

    TrackletLUT*& phiCorr(unsigned int layer) { return thePhiCorr_[layer]; }

    std::map<std::string, std::vector<int> >& ILindex() { return ILindex_; }

    std::map<std::string, int>& layerdiskmap() { return layerdiskmap_; }

    IMATH_TrackletCalculator* ITC_L1L2() { return ITC_L1L2_.get(); }
    IMATH_TrackletCalculator* ITC_L2L3() { return ITC_L2L3_.get(); }
    IMATH_TrackletCalculator* ITC_L3L4() { return ITC_L3L4_.get(); }
    IMATH_TrackletCalculator* ITC_L5L6() { return ITC_L5L6_.get(); }

    IMATH_TrackletCalculatorDisk* ITC_F1F2() { return ITC_F1F2_.get(); }
    IMATH_TrackletCalculatorDisk* ITC_F3F4() { return ITC_F3F4_.get(); }
    IMATH_TrackletCalculatorDisk* ITC_B1B2() { return ITC_B1B2_.get(); }
    IMATH_TrackletCalculatorDisk* ITC_B3B4() { return ITC_B3B4_.get(); }

    IMATH_TrackletCalculatorOverlap* ITC_L1F1() { return ITC_L1F1_.get(); }
    IMATH_TrackletCalculatorOverlap* ITC_L1B1() { return ITC_L1B1_.get(); }
    IMATH_TrackletCalculatorOverlap* ITC_L2F1() { return ITC_L2F1_.get(); }
    IMATH_TrackletCalculatorOverlap* ITC_L2B1() { return ITC_L2B1_.get(); }

    std::ofstream& ofstream(std::string fname);

#ifdef USEHYBRID
    std::unique_ptr<tmtt::Settings>& tmttSettings() { return tmttSettings_; }
    std::unique_ptr<tmtt::KFParamsComb>& tmttKFParamsComb() { return tmttKFParamsComb_; }
#endif

  private:
    std::unordered_map<std::string, std::ofstream*> ofstreams_;

    std::unique_ptr<imathGlobals> imathGlobals_;

    // tracklet calculators
    std::unique_ptr<IMATH_TrackletCalculator> ITC_L1L2_;
    std::unique_ptr<IMATH_TrackletCalculator> ITC_L2L3_;
    std::unique_ptr<IMATH_TrackletCalculator> ITC_L3L4_;
    std::unique_ptr<IMATH_TrackletCalculator> ITC_L5L6_;

    std::unique_ptr<IMATH_TrackletCalculatorDisk> ITC_F1F2_;
    std::unique_ptr<IMATH_TrackletCalculatorDisk> ITC_F3F4_;
    std::unique_ptr<IMATH_TrackletCalculatorDisk> ITC_B1B2_;
    std::unique_ptr<IMATH_TrackletCalculatorDisk> ITC_B3B4_;

    std::unique_ptr<IMATH_TrackletCalculatorOverlap> ITC_L1F1_;
    std::unique_ptr<IMATH_TrackletCalculatorOverlap> ITC_L2F1_;
    std::unique_ptr<IMATH_TrackletCalculatorOverlap> ITC_L1B1_;
    std::unique_ptr<IMATH_TrackletCalculatorOverlap> ITC_L2B1_;

    SLHCEvent* theEvent_{nullptr};

    HistBase* theHistBase_{nullptr};

    TrackDerTable* trackDerTable_{nullptr};

#ifdef USEHYBRID
    std::unique_ptr<tmtt::Settings> tmttSettings_;
    std::unique_ptr<tmtt::KFParamsComb> tmttKFParamsComb_;
#endif

    std::array<TrackletLUT*, 6> thePhiCorr_{{nullptr, nullptr, nullptr, nullptr, nullptr, nullptr}};

    std::map<std::string, std::vector<int> > ILindex_;

    std::map<std::string, int> layerdiskmap_;
  };
};  // namespace trklet

#endif
