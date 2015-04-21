#ifndef TrackerTrackerValidationVariables_h
#define TrackerTrackerValidationVariables_h

// system include files
#include <vector>

#include "FWCore/Utilities/interface/EDGetToken.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

class MagneticField;
class Trajectory;

namespace edm {
  class ConsumesCollector;
  class Event;
  class EventSetup;
  class ParameterSet;
}

class TrackerValidationVariables
{ 
 public:  
 
  struct AVHitStruct
  {
  AVHitStruct() : resX(-999.), resY(-999.), resErrX(-999.), resErrY(-999.), resXprime(-999.), resXatTrkY(-999.), resXprimeErr(-999.),  
      resYprime(-999.), resYprimeErr(-999.), phi(-999.), eta(-999.),
     inside(false), localX(-999.), localY(-999.), localXnorm(-999.), localYnorm(-999.), localAlpha(-999.), localBeta(-999.),
      rawDetId(0) {}
    float resX;
    float resY;
    float resErrX;
    float resErrY;
    float resXprime;
    float resXatTrkY;
    float resXprimeErr;
    float resYprime;
    float resYprimeErr;
    float phi;
    float eta;
    bool inside;
    float localX;
    float localY;
    float localXnorm;
    float localYnorm;
    float localAlpha;    
    float localBeta;
    uint32_t rawDetId;
  };
  
  struct AVTrackStruct
  {
  AVTrackStruct() : p(0.), pt(0.), ptError(0.), px(0.), py(0.), pz(0.), eta(0.), phi(0.), kappa(0.),
      chi2(0.), chi2Prob(0.), normchi2(0), d0(-999.), dz(-999.), charge(-999), numberOfValidHits(0), numberOfLostHits(0) {};
    float p;
    float pt;
    float ptError;
    float px;
    float py;
    float pz;
    float eta;
    float phi;
    float kappa;
    float chi2;
    float chi2Prob;
    float normchi2;
    float d0;
    float dz;
    int charge;
    int numberOfValidHits;
    int numberOfLostHits;
    std::vector<AVHitStruct> hits;
  };

  TrackerValidationVariables();
  TrackerValidationVariables(const edm::ParameterSet& config,
                             edm::ConsumesCollector && iC);
  ~TrackerValidationVariables();

  void fillHitQuantities(const Trajectory* trajectory, std::vector<AVHitStruct> & v_avhitout);
  void fillTrackQuantities(const edm::Event&,
                           const edm::EventSetup&,
                           std::vector<AVTrackStruct> & v_avtrackout);

  // need the following method for MonitorTrackResiduals in DQM/TrackerMonitorTrack
  void fillHitQuantities(const edm::Event&, std::vector<AVHitStruct> & v_avhitout);

 private:

  edm::EDGetTokenT<std::vector<Trajectory> > trajCollectionToken_;
  edm::EDGetTokenT<TrajTrackAssociationCollection> trajTracksToken_;
};

#endif
