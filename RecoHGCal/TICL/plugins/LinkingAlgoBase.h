#ifndef RecoHGCal_TICL_LinkingAlgoBase_H__
#define RecoHGCal_TICL_LinkingAlgoBase_H__

#include <vector>
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/HGCalReco/interface/TICLCandidate.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"

namespace edm {
  class Event;
  class EventSetup;
}  // namespace edm

namespace ticl {
  class LinkingAlgoBase {
  public:
    LinkingAlgoBase(const edm::ParameterSet& conf) : algo_verbosity_(conf.getParameter<int>("algo_verbosity")) {}

    virtual ~LinkingAlgoBase(){};

    virtual void initialize(const HGCalDDDConstants* hgcons,
                            const hgcal::RecHitTools rhtools,
                            const edm::ESHandle<MagneticField> bfieldH,
                            const edm::ESHandle<Propagator> propH) = 0;

    virtual void linkTracksters(const edm::Handle<std::vector<reco::Track>> tkH,
                                const edm::ValueMap<float>& tkTime,
                                const edm::ValueMap<float>& tkTimeErr,
                                const edm::ValueMap<float>& tkTimeQual,
                                const std::vector<reco::Muon>& muons,
                                const edm::Handle<std::vector<Trackster>> tsH,
                                std::vector<TICLCandidate>& resultTracksters,
                                std::vector<TICLCandidate>& resultFromTracks) = 0;

    static void fillPSetDescription(edm::ParameterSetDescription& desc) { desc.add<int>("algo_verbosity", 0); };

  protected:
    int algo_verbosity_;
  };
}  // namespace ticl

#endif
