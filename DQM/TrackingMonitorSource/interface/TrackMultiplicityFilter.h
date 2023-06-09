#ifndef DQM_TrackingMonitorSource_TrackMultiplicityFilter_H
#define DQM_TrackingMonitorSource_TrackMultiplicityFilter_H

// system include files
#include <memory>
#include <algorithm>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
//
// class declaration
//

class TrackMultiplicityFilter : public edm::global::EDFilter<> {
public:
  explicit TrackMultiplicityFilter(const edm::ParameterSet&);
  ~TrackMultiplicityFilter() override = default;

private:
  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  // ----------member data ---------------------------
  const edm::InputTag tracksTag_;
  edm::EDGetTokenT<reco::TrackCollection> tracksToken_;

  StringCutObjectSelector<reco::Track> selector_;

  unsigned int nmin_;
};
#endif
