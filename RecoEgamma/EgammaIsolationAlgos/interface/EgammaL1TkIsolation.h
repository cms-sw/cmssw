#ifndef RECOEGAMMA_EGAMMAISOLATIONALGOS_EGAMMAL1TKISOLATION_H
#define RECOEGAMMA_EGAMMAISOLATIONALGOS_EGAMMAL1TKISOLATION_H

#include "DataFormats/L1TrackTrigger/interface/L1Track.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

//author S. Harper (RAL/CERN) 
//based on the work of Swagata Mukherjee and Giulia Sorrentino

class EgammaL1TkIsolation {
public:

private:
  struct TrkCuts {
    float minPt;
    float minDR2;
    float maxDR2;
    float minDEta;
    float maxDZ;
    explicit TrkCuts(const edm::ParameterSet& para);
    static edm::ParameterSetDescription pSetDescript();
  };

  TrkCuts barrelCuts_, endcapCuts_;

public:
  explicit EgammaL1TkIsolation(const edm::ParameterSet& para);
  EgammaL1TkIsolation(const EgammaL1TkIsolation&) = default;
  ~EgammaL1TkIsolation() = default;
  EgammaL1TkIsolation& operator=(const EgammaL1TkIsolation&) = default;

  static edm::ParameterSetDescription pSetDescript();

  std::pair<int, double> calIsol(const reco::TrackBase& trk,
				 const L1TrackCollection& l1Tks)const;

  std::pair<int, double> calIsol(const double objEta,
                                 const double objPhi,
                                 const double objZ,
                                 const L1TrackCollection& l1Tks)const;

  //little helper function for the two calIsol functions for it to directly return the pt
  template <typename... Args>
  double calIsolPt(Args&&... args) const {
    return calIsol(std::forward<Args>(args)...).second;
  }


private:
  static bool passTrkSel(const L1Track& trk,
                         const double trkPt,
                         const TrkCuts& cuts,
                         const double objEta,
                         const double objPhi,
                         const double objZ);
};

#endif
