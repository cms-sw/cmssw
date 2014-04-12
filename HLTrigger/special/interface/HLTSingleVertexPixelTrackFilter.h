
#ifndef HLTSingleVertexPixelTrackFilter_h
#define HLTSingleVertexPixelTrackFilter_h

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"


namespace edm {
  class ConfigurationDescriptions;
}
//
// class declaration
//

class HLTSingleVertexPixelTrackFilter : public HLTFilter {

   public:
      explicit HLTSingleVertexPixelTrackFilter(const edm::ParameterSet&);
      ~HLTSingleVertexPixelTrackFilter();
      virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

   private:
      edm::InputTag pixelVerticesTag_;  // input tag identifying product containing Pixel-vertices
      edm::InputTag pixelTracksTag_;  // input tag identifying product containing Pixel-tracks
      edm::EDGetTokenT<reco::VertexCollection> pixelVerticesToken_;
      edm::EDGetTokenT<reco::RecoChargedCandidateCollection> pixelTracksToken_;

      double min_Pt_;          // min pt cut
      double max_Pt_;          // max pt cut
      double max_Eta_;          // max eta cut
      double max_Vz_;          // max vz cut
      int min_trks_;  // minimum number of tracks from one vertex
      float min_sep_;          // minimum separation of two tracks in phi-eta
};

#endif //HLTSingleVertexPixelTrackFilter_h
