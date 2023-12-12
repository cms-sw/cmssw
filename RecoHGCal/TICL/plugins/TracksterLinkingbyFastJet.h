#ifndef RecoHGCal_TICL_TracksterLinkingAlgoByFastJet_H
#define RecoHGCal_TICL_TracksterLinkingAlgoByFastJet_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "RecoHGCal/TICL/interface/TracksterLinkingAlgoBase.h"
#include "fastjet/ClusterSequence.hh"
#include "DataFormats/Math/interface/deltaR.h"

namespace ticl {

class TracksterLinkingbyFastJet : public TracksterLinkingAlgoBase {
public:
  TracksterLinkingbyFastJet(const edm::ParameterSet& conf, edm::ConsumesCollector iC)
      : TracksterLinkingAlgoBase(conf, iC),
        antikt_radius_(conf.getParameter<double>("antikt_radius")) {}
  virtual ~TracksterLinkingbyFastJet() {}

  void linkTracksters(const Inputs& input, std::vector<Trackster>& resultTracksters,
                      std::vector<std::vector<unsigned int>>& linkedResultTracksters,
                      std::vector<std::vector<unsigned int>>& linkedTracksterIdToInputTracksterId) override;
  
private:
  float antikt_radius_;
};

}  // namespace ticl

#endif
