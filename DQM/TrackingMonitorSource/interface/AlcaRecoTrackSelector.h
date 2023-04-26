#ifndef DQM_TrackingMonitorSource_ALCARECOTRACKSELECTOR_H
#define DQM_TrackingMonitorSource_ALCARECOTRACKSELECTOR_H

// system include files
#include <memory>
#include <algorithm>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

//                                                                        // class declaration                                                    //

class AlcaRecoTrackSelector : public edm::one::EDProducer<> {
public:
  explicit AlcaRecoTrackSelector(const edm::ParameterSet&);
  ~AlcaRecoTrackSelector() override = default;

private:
  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup) override;

  edm::ParameterSet parameters_;
  const edm::InputTag tracksTag_;
  const edm::EDGetTokenT<reco::TrackCollection> tracksToken_;
  const double ptmin_;
  const double pmin_;
  const double etamin_;
  const double etamax_;
  const int nhits_;
};
#endif
