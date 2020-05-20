// Globals: holds "global" variables such as the IMATH_TrackletCalculators
#ifndef L1Trigger_TrackFindingTracklet_interface_Globals_h
#define L1Trigger_TrackFindingTracklet_interface_Globals_h

#include <memory>

#include "HistBase.h"

#include "L1Trigger/TrackFindingTracklet/interface/IMATH_TrackletCalculator.h"
#include "L1Trigger/TrackFindingTracklet/interface/IMATH_TrackletCalculatorDisk.h"
#include "L1Trigger/TrackFindingTracklet/interface/IMATH_TrackletCalculatorOverlap.h"
#include "L1Trigger/TrackFindingTracklet/interface/VMRouterPhiCorrTable.h"

namespace tmtt {
  class Settings;
  class KFParamsComb;
}  // namespace tmtt

namespace trklet {

  class TETableBase;
  class TrackDerTable;
  class ProjectionRouterBendTable;
  class SLHCEvent;

  class Globals {
  public:
    Globals(const Settings* settings) {
      imathGlobals* imathGlobs = new imathGlobals();

      //takes owernship of globals pointer
      imathGlobals_.reset(imathGlobs);

      // tracklet calculators
      ITC_L1L2_.reset(new IMATH_TrackletCalculator(settings, imathGlobs, 1, 2));
      ITC_L2L3_.reset(new IMATH_TrackletCalculator(settings, imathGlobs, 2, 3));
      ITC_L3L4_.reset(new IMATH_TrackletCalculator(settings, imathGlobs, 3, 4));
      ITC_L5L6_.reset(new IMATH_TrackletCalculator(settings, imathGlobs, 5, 6));

      ITC_F1F2_.reset(new IMATH_TrackletCalculatorDisk(settings, imathGlobs, 1, 2));
      ITC_F3F4_.reset(new IMATH_TrackletCalculatorDisk(settings, imathGlobs, 3, 4));
      ITC_B1B2_.reset(new IMATH_TrackletCalculatorDisk(settings, imathGlobs, -1, -2));
      ITC_B3B4_.reset(new IMATH_TrackletCalculatorDisk(settings, imathGlobs, -3, -4));

      ITC_L1F1_.reset(new IMATH_TrackletCalculatorOverlap(settings, imathGlobs, 1, 1));
      ITC_L2F1_.reset(new IMATH_TrackletCalculatorOverlap(settings, imathGlobs, 2, 1));
      ITC_L1B1_.reset(new IMATH_TrackletCalculatorOverlap(settings, imathGlobs, 1, -1));
      ITC_L2B1_.reset(new IMATH_TrackletCalculatorOverlap(settings, imathGlobs, 2, -1));
    }

    ~Globals() {
      for (auto i : thePhiCorr_) {
        delete i;
      }
    };

    SLHCEvent*& event() { return theEvent_; }

    HistBase*& histograms() { return theHistBase_; }

    TrackDerTable*& trackDerTable() { return trackDerTable_; }

    tmtt::Settings*& tmttSettings() { return tmttSettings_; }

    tmtt::KFParamsComb*& tmttKFParamsComb() { return tmttKFParamsComb_; }

    VMRouterPhiCorrTable*& phiCorr(unsigned int layer) { return thePhiCorr_[layer]; }

    ProjectionRouterBendTable*& projectionRouterBendTable() { return projectionRouterBendTable_; }

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

    std::ofstream& ofstream(std::string fname) {
      if (ofstreams_.find(fname) != ofstreams_.end()) {
        return *(ofstreams_[fname]);
      }
      std::ofstream* outptr = new std::ofstream(fname.c_str());
      ofstreams_[fname] = outptr;
      return *outptr;
    }

  private:
    std::map<std::string, std::ofstream*> ofstreams_;

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

    ProjectionRouterBendTable* projectionRouterBendTable_{nullptr};

    tmtt::Settings* tmttSettings_{nullptr};

    tmtt::KFParamsComb* tmttKFParamsComb_{nullptr};

    std::array<VMRouterPhiCorrTable*, 6> thePhiCorr_{{nullptr, nullptr, nullptr, nullptr, nullptr, nullptr}};

    std::map<std::string, std::vector<int> > ILindex_;

    std::map<std::string, int> layerdiskmap_;
  };
};  // namespace trklet

#endif
