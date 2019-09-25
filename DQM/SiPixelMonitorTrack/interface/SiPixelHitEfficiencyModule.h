// Package:    SiPixelMonitorTrack
// Class:      SiPixelHitEfficiencyModule
//
// class SiPixelHitEfficiencyModule SiPixelHitEfficiencyModule.h
//       DQM/SiPixelMonitorTrack/src/SiPixelHitEfficiencyModule.h
//
// Description: SiPixel hit efficiency data quality monitoring modules
// Implementation: prototype -> improved -> never final - end of the 1st step
//
// Original Authors: Romain Rougny & Luca Mucibello
//         Created: Mar Nov 10 13:29:00 CET 2009

#ifndef SiPixelMonitorTrack_SiPixelHitEfficiencyModule_h
#define SiPixelMonitorTrack_SiPixelHitEfficiencyModule_h

#include <utility>

//#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementVector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h"
#include <cstdint>

namespace edm {
  class EventSetup;
}

class SiPixelHitEfficiencyModule {
public:
  typedef dqm::reco::DQMStore DQMStore;
  typedef dqm::reco::MonitorElement MonitorElement;

  SiPixelHitEfficiencyModule();
  SiPixelHitEfficiencyModule(const uint32_t);
  ~SiPixelHitEfficiencyModule();

  void book(
      const edm::ParameterSet &, edm::EventSetup const &, DQMStore::IBooker &, int type = 0, bool isUpgrade = false);
  void fill(const TrackerTopology *pTT,
            const LocalTrajectoryParameters &ltp,
            bool isHitValid,
            bool modon = true,
            bool ladon = true,
            bool layon = true,
            bool phion = true,
            bool bladeon = true,
            bool diskon = true,
            bool ringon = true);
  void computeEfficiencies(bool modon = true,
                           bool ladon = true,
                           bool layon = true,
                           bool phion = true,
                           bool bladeon = true,
                           bool diskon = true,
                           bool ringon = true);
  std::pair<double, double> eff(double nValid, double nMissing);

private:
  uint32_t id_;
  bool bBookTracks;
  bool debug_;
  bool updateEfficiencies;

  // EFFICIENCY
  MonitorElement *meEfficiency_;
  MonitorElement *meEfficiencyX_;
  MonitorElement *meEfficiencyY_;
  MonitorElement *meEfficiencyAlpha_;
  MonitorElement *meEfficiencyBeta_;

  MonitorElement *meEfficiencyLad_;
  MonitorElement *meEfficiencyXLad_;
  MonitorElement *meEfficiencyYLad_;
  MonitorElement *meEfficiencyAlphaLad_;
  MonitorElement *meEfficiencyBetaLad_;

  MonitorElement *meEfficiencyLay_;
  MonitorElement *meEfficiencyXLay_;
  MonitorElement *meEfficiencyYLay_;
  MonitorElement *meEfficiencyAlphaLay_;
  MonitorElement *meEfficiencyBetaLay_;

  MonitorElement *meEfficiencyPhi_;
  MonitorElement *meEfficiencyXPhi_;
  MonitorElement *meEfficiencyYPhi_;
  MonitorElement *meEfficiencyAlphaPhi_;
  MonitorElement *meEfficiencyBetaPhi_;

  MonitorElement *meEfficiencyBlade_;
  MonitorElement *meEfficiencyXBlade_;
  MonitorElement *meEfficiencyYBlade_;
  MonitorElement *meEfficiencyAlphaBlade_;
  MonitorElement *meEfficiencyBetaBlade_;

  MonitorElement *meEfficiencyDisk_;
  MonitorElement *meEfficiencyXDisk_;
  MonitorElement *meEfficiencyYDisk_;
  MonitorElement *meEfficiencyAlphaDisk_;
  MonitorElement *meEfficiencyBetaDisk_;

  MonitorElement *meEfficiencyRing_;
  MonitorElement *meEfficiencyXRing_;
  MonitorElement *meEfficiencyYRing_;
  MonitorElement *meEfficiencyAlphaRing_;
  MonitorElement *meEfficiencyBetaRing_;

  // VALID HITS
  MonitorElement *meValid_;
  MonitorElement *meValidX_;
  MonitorElement *meValidY_;
  MonitorElement *meValidAlpha_;
  MonitorElement *meValidBeta_;

  MonitorElement *meValidLad_;
  MonitorElement *meValidXLad_;
  MonitorElement *meValidYLad_;
  MonitorElement *meValidModLad_;
  MonitorElement *meValidAlphaLad_;
  MonitorElement *meValidBetaLad_;

  MonitorElement *meValidLay_;
  MonitorElement *meValidXLay_;
  MonitorElement *meValidYLay_;
  MonitorElement *meValidAlphaLay_;
  MonitorElement *meValidBetaLay_;

  MonitorElement *meValidPhi_;
  MonitorElement *meValidXPhi_;
  MonitorElement *meValidYPhi_;
  MonitorElement *meValidAlphaPhi_;
  MonitorElement *meValidBetaPhi_;

  MonitorElement *meValidBlade_;
  MonitorElement *meValidXBlade_;
  MonitorElement *meValidYBlade_;
  MonitorElement *meValidAlphaBlade_;
  MonitorElement *meValidBetaBlade_;

  MonitorElement *meValidDisk_;
  MonitorElement *meValidXDisk_;
  MonitorElement *meValidYDisk_;
  MonitorElement *meValidAlphaDisk_;
  MonitorElement *meValidBetaDisk_;

  MonitorElement *meValidRing_;
  MonitorElement *meValidXRing_;
  MonitorElement *meValidYRing_;
  MonitorElement *meValidAlphaRing_;
  MonitorElement *meValidBetaRing_;

  // MISSING HITS
  MonitorElement *meMissing_;
  MonitorElement *meMissingX_;
  MonitorElement *meMissingY_;
  MonitorElement *meMissingAlpha_;
  MonitorElement *meMissingBeta_;

  MonitorElement *meMissingLad_;
  MonitorElement *meMissingXLad_;
  MonitorElement *meMissingYLad_;
  MonitorElement *meMissingModLad_;
  MonitorElement *meMissingAlphaLad_;
  MonitorElement *meMissingBetaLad_;

  MonitorElement *meMissingLay_;
  MonitorElement *meMissingXLay_;
  MonitorElement *meMissingYLay_;
  MonitorElement *meMissingAlphaLay_;
  MonitorElement *meMissingBetaLay_;

  MonitorElement *meMissingPhi_;
  MonitorElement *meMissingXPhi_;
  MonitorElement *meMissingYPhi_;
  MonitorElement *meMissingAlphaPhi_;
  MonitorElement *meMissingBetaPhi_;

  MonitorElement *meMissingBlade_;
  MonitorElement *meMissingXBlade_;
  MonitorElement *meMissingYBlade_;
  MonitorElement *meMissingAlphaBlade_;
  MonitorElement *meMissingBetaBlade_;

  MonitorElement *meMissingDisk_;
  MonitorElement *meMissingXDisk_;
  MonitorElement *meMissingYDisk_;
  MonitorElement *meMissingAlphaDisk_;
  MonitorElement *meMissingBetaDisk_;

  MonitorElement *meMissingRing_;
  MonitorElement *meMissingXRing_;
  MonitorElement *meMissingYRing_;
  MonitorElement *meMissingAlphaRing_;
  MonitorElement *meMissingBetaRing_;
};

#endif
