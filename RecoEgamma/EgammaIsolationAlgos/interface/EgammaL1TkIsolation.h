#ifndef RecoEgamma_EgammaIsolationAlgos_EgammaL1TkIsolation_h
#define RecoEgamma_EgammaIsolationAlgos_EgammaL1TkIsolation_h

#include "DataFormats/L1TrackTrigger/interface/L1Track.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

//author S. Harper (RAL/CERN)
//based on the work of Swagata Mukherjee and Giulia Sorrentino

class EgammaL1TkIsolation {
public:
  explicit EgammaL1TkIsolation(const edm::ParameterSet& para);

  static void fillPSetDescription(edm::ParameterSetDescription& desc);
  static edm::ParameterSetDescription makePSetDescription() {
    edm::ParameterSetDescription desc;
    fillPSetDescription(desc);
    return desc;
  }

  std::pair<int, double> calIsol(const reco::TrackBase& trk, const L1TrackCollection& l1Tks) const;

  std::pair<int, double> calIsol(const double objEta,
                                 const double objPhi,
                                 const double objZ,
                                 const L1TrackCollection& l1Tks) const;

  //little helper function for the two calIsol functions for it to directly return the pt
  template <typename... Args>
  double calIsolPt(Args&&... args) const {
    return calIsol(std::forward<Args>(args)...).second;
  }

private:
  struct TrkCuts {
    float minPt;
    float minDR2;
    float maxDR2;
    float minDEta;
    float maxDZ;
    explicit TrkCuts(const edm::ParameterSet& para);
    static edm::ParameterSetDescription makePSetDescription();
  };

  size_t etaBinNr(double eta) const;
  static bool passTrkSel(const L1Track& trk,
                         const double trkPt,
                         const TrkCuts& cuts,
                         const double objEta,
                         const double objPhi,
                         const double objZ);

  bool useAbsEta_;
  std::vector<double> etaBoundaries_;
  std::vector<TrkCuts> trkCuts_;
};

#endif
