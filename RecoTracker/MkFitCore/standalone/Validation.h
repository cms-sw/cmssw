#ifndef RecoTracker_MkFitCore_standalone_Validation_h
#define RecoTracker_MkFitCore_standalone_Validation_h

#include "RecoTracker/MkFitCore/interface/Track.h"
#include "RecoTracker/MkFitCore/standalone/ConfigStandalone.h"
#include "RecoTracker/MkFitCore/standalone/TrackExtra.h"

namespace mkfit {

  class Event;

  // Fit Validation objects -- mplex only
  struct FitVal {
  public:
    FitVal() {}
    FitVal(float ppz,
           float eppz,
           float ppphi,
           float eppphi,
           float upt,
           float eupt,
           float umphi,
           float eumphi,
           float umeta,
           float eumeta)
        : ppz(ppz),
          eppz(eppz),
          ppphi(ppphi),
          eppphi(eppphi),
          upt(upt),
          eupt(eupt),
          umphi(umphi),
          eumphi(eumphi),
          umeta(umeta),
          eumeta(eumeta) {}

    // first p or u = propagated or updated
    // middle: p or m/nothing = position or momentum
    // begining: e = error (already sqrt)
    float ppz, eppz, ppphi, eppphi;
    float upt, eupt, umphi, eumphi, umeta, eumeta;
  };

  class Validation {
  public:
    virtual ~Validation() {}

    virtual void alignTracks(TrackVec&, TrackExtraVec&, bool) {}

    virtual void resetValidationMaps() {}
    virtual void resetDebugVectors() {}

    virtual void collectFitInfo(const FitVal&, int, int) {}

    virtual void setTrackExtras(Event& ev) {}
    virtual void makeSimTkToRecoTksMaps(Event&) {}
    virtual void makeSeedTkToRecoTkMaps(Event&) {}
    virtual void makeRecoTkToRecoTkMaps(Event&) {}
    virtual void makeCMSSWTkToRecoTksMaps(Event&) {}
    virtual void makeSeedTkToCMSSWTkMap(Event&) {}
    virtual void makeCMSSWTkToSeedTkMap(Event&) {}
    virtual void makeRecoTkToSeedTkMapsDumbCMSSW(Event&) {}

    virtual void setTrackScoresDumbCMSSW(Event&) {}

    virtual void fillEfficiencyTree(const Event&) {}
    virtual void fillFakeRateTree(const Event&) {}
    virtual void fillConfigTree() {}
    virtual void fillCMSSWEfficiencyTree(const Event&) {}
    virtual void fillCMSSWFakeRateTree(const Event&) {}
    virtual void fillFitTree(const Event&) {}

    virtual void saveTTrees() {}

    static Validation* make_validation(const std::string&, const TrackerInfo*);

  protected:
    Validation();
  };

}  // end namespace mkfit
#endif
