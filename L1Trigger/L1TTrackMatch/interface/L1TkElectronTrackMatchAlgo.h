#ifndef L1Trigger_L1TTrackMatch_L1TkElectronTrackMatchAlgo_HH
#define L1Trigger_L1TTrackMatch_L1TkElectronTrackMatchAlgo_HH

#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

namespace L1TkElectronTrackMatchAlgo {
  typedef TTTrack<Ref_Phase2TrackerDigi_> L1TTTrackType;
  typedef std::vector<L1TTTrackType> L1TTTrackCollection;
  void doMatch(BXVector<l1t::EGamma>::const_iterator egIter,
               const edm::Ptr<L1TTTrackType>& pTrk,
               double& dph,
               double& dr,
               double& deta);
  void doMatchClusterET(BXVector<l1t::EGamma>::const_iterator egIter,
                        const edm::Ptr<L1TTTrackType>& pTrk,
                        double& dph,
                        double& dr,
                        double& deta);
  void doMatch(const GlobalPoint& epos, const edm::Ptr<L1TTTrackType>& pTrk, double& dph, double& dr, double& deta);

  double deltaR(const GlobalPoint& epos, const edm::Ptr<L1TTTrackType>& pTrk);
  double deltaPhi(const GlobalPoint& epos, const edm::Ptr<L1TTTrackType>& pTrk);
  double deltaPhiClusterET(BXVector<l1t::EGamma>::const_iterator egIter, const edm::Ptr<L1TTTrackType>& pTrk);
  double deltaEta(const GlobalPoint& epos, const edm::Ptr<L1TTTrackType>& pTrk);
  GlobalPoint calorimeterPosition(double phi, double eta, double e);

}  // namespace L1TkElectronTrackMatchAlgo
#endif
