#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
namespace PFTrackAlgoTools {

  double dPtCut(const reco::TrackBase::TrackAlgorithm&, const std::vector<double>&, bool);
  unsigned int nHitCut(const reco::TrackBase::TrackAlgorithm&, const std::vector<unsigned int>&, bool);
  double errorScale(const reco::TrackBase::TrackAlgorithm&, const std::vector<double>&);
  bool isGoodForEGM(const reco::TrackBase::TrackAlgorithm&);
  bool isGoodForEGMPrimary(const reco::TrackBase::TrackAlgorithm&);
  bool isFifthStep(const reco::TrackBase::TrackAlgorithm&);

  bool nonIterative(const reco::TrackBase::TrackAlgorithm&);
  bool highQuality(const reco::TrackBase::TrackAlgorithm&);
  bool step45(const reco::TrackBase::TrackAlgorithm&);
  bool step5(const reco::TrackBase::TrackAlgorithm&);

  bool goodPtResolution(const reco::TrackRef&,
                        const std::vector<double>& DPtovPtCut,
                        const std::vector<unsigned>& NHitCut,
                        bool useIterTracking,
                        //const reco::TrackBase::TrackQuality trackQuality = reco::TrackBase::highPurity,
                        const reco::TrackBase::TrackQuality trackQuality);
}  // namespace PFTrackAlgoTools
