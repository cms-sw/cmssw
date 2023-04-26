#ifndef TRACKMULTIPLICITYFILTER_H
#define TRACKMULTIPLICITYFILTER_H

// system include files
#include <memory>
#include <algorithm>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

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

  //  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  // ----------member data ---------------------------
  edm::ParameterSet parameters_;
  const edm::InputTag tracksTag_;
  edm::EDGetTokenT<reco::TrackCollection> tracksToken_;

  StringCutObjectSelector<reco::Track> selector_;

  unsigned int nmin_;
};
#endif
