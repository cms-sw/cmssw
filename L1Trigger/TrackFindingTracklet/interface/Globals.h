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

    std::ofstream& ofstream(std::string fname);

#ifdef USEHYBRID
    std::unique_ptr<tmtt::Settings>& tmttSettings() { return tmttSettings_; }
    std::unique_ptr<tmtt::KFParamsComb>& tmttKFParamsComb() { return tmttKFParamsComb_; }
#endif

  private:
    std::unordered_map<std::string, std::ofstream*> ofstreams_;

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
