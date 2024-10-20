#ifndef RecoEgamma_EgammaIsolationAlgos_SuperclusTkIsolFromCands_h
#define RecoEgamma_EgammaIsolationAlgos_SuperclusTkIsolFromCands_h

#include "CommonTools/Utils/interface/KinematicColumns.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "FWCore/SOA/interface/Table.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EleTkIsolFromCands.h"

class SuperclusTkIsolFromCands : public EleTkIsolFromCands {
public:
  explicit SuperclusTkIsolFromCands(Configuration const& cfg, reco::TrackCollection const& tracks)
      : EleTkIsolFromCands(cfg, tracks) {}
  explicit SuperclusTkIsolFromCands(Configuration const& cfg,
                                    pat::PackedCandidateCollection const& cands,
                                    PIDVeto pidVeto = PIDVeto::NONE)
      : EleTkIsolFromCands(cfg, cands, pidVeto) {}

  Output operator()(const reco::SuperCluster& sc, const math::XYZPoint& vtx);
};

#endif
