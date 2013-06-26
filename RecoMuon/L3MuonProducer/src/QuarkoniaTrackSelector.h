#ifndef QuarkoniaTrackSelector_h_
#define QuarkoniaTrackSelector_h_
/** Creates a filtered TrackCollection based on the mass of a combination 
 *  of a track and a RecoChargedCandidate (typically a muon)
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <vector>


class QuarkoniaTrackSelector : public edm::EDProducer {
public:
  explicit QuarkoniaTrackSelector(const edm::ParameterSet&);
  ~QuarkoniaTrackSelector() {}

private:
  virtual void produce(edm::Event&, const edm::EventSetup&);
      
private:
  edm::InputTag muonTag_;          ///< tag for RecoChargedCandidateCollection
  edm::InputTag trackTag_;         ///< tag for TrackCollection
  std::vector<double> minMasses_;  ///< lower mass limits
  std::vector<double> maxMasses_;  ///< upper mass limits
  bool checkCharge_;               ///< check opposite charge?
  double minTrackPt_;              ///< track pt cut
  double minTrackP_;               ///< track p cut
  double maxTrackEta_;             ///< track |eta| cut
};

#endif
