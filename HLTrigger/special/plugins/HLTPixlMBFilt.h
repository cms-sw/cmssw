#ifndef HLTPixlMBFilt_h
#define HLTPixlMBFilt_h

/** \class HLTFiltCand
 *
 *
 *  This class is an HLTFilter (-> EDFilter) implementing a minimum-bias
 *  HLT trigger acting on candidates, requiring tracks in Pixel det
 *
 *
 *  \author Mika Huhtinen
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"

namespace edm {
  class ConfigurationDescriptions;
}

//
// class declaration
//

class HLTPixlMBFilt : public HLTFilter {
public:
  explicit HLTPixlMBFilt(const edm::ParameterSet&);
  ~HLTPixlMBFilt() override;
  bool hltFilter(edm::Event&,
                 const edm::EventSetup&,
                 trigger::TriggerFilterObjectWithRefs& filterproduct) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::InputTag pixlTag_;  // input tag identifying product containing Pixel-tracks
  edm::EDGetTokenT<reco::RecoChargedCandidateCollection> pixlToken_;

  double min_Pt_;          // min pt cut
  unsigned int min_trks_;  // minimum number of tracks from one vertex
  float min_sep_;          // minimum separation of two tracks in phi-eta
};

#endif  //HLTPixlMBFilt_h
