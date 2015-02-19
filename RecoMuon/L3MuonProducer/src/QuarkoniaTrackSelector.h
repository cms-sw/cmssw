#ifndef QuarkoniaTrackSelector_h_
#define QuarkoniaTrackSelector_h_
/** Creates a filtered TrackCollection based on the mass of a combination 
 *  of a track and a RecoChargedCandidate (typically a muon)
 */

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include <vector>


class QuarkoniaTrackSelector : public edm::global::EDProducer<> {
public:
  explicit QuarkoniaTrackSelector(const edm::ParameterSet&);
  ~QuarkoniaTrackSelector() {}

private:
  virtual void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
      
private:
  edm::InputTag muonTag_;          ///< tag for RecoChargedCandidateCollection
  edm::InputTag trackTag_;         ///< tag for TrackCollection
  edm::EDGetTokenT<reco::RecoChargedCandidateCollection> muonToken_;
  edm::EDGetTokenT<reco::TrackCollection> trackToken_;

  std::vector<double> minMasses_;  ///< lower mass limits
  std::vector<double> maxMasses_;  ///< upper mass limits
  bool checkCharge_;               ///< check opposite charge?
  double minTrackPt_;              ///< track pt cut
  double minTrackP_;               ///< track p cut
  double maxTrackEta_;             ///< track |eta| cut
};

#endif
