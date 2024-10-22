#include "HLTPixelIsolTrackL1TFilter.h"

#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

HLTPixelIsolTrackL1TFilter::HLTPixelIsolTrackL1TFilter(const edm::ParameterSet& iConfig)
    : HLTFilter(iConfig),
      candTag_{iConfig.getParameter<edm::InputTag>("candTag")},
      candToken_{consumes(candTag_)},
      maxptnearby_{iConfig.getParameter<double>("MaxPtNearby")},
      minEnergy_{iConfig.getParameter<double>("MinEnergyTrack")},
      minpttrack_{iConfig.getParameter<double>("MinPtTrack")},
      maxetatrack_{iConfig.getParameter<double>("MaxEtaTrack")},
      minetatrack_{iConfig.getParameter<double>("MinEtaTrack")},
      filterE_{iConfig.getParameter<bool>("filterTrackEnergy")},
      nMaxTrackCandidates_{iConfig.getParameter<int>("NMaxTrackCandidates")},
      dropMultiL2Event_{iConfig.getParameter<bool>("DropMultiL2Event")} {}

void HLTPixelIsolTrackL1TFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("candTag", edm::InputTag("hltIsolPixelTrackProd"));
  desc.add<double>("MaxPtNearby", 2.0);
  desc.add<double>("MinEnergyTrack", 12.0);
  desc.add<double>("MinPtTrack", 3.5);
  desc.add<double>("MaxEtaTrack", 1.15);
  desc.add<double>("MinEtaTrack", 0.0);
  desc.add<bool>("filterTrackEnergy", true);
  desc.add<int>("NMaxTrackCandidates", 10);
  desc.add<bool>("DropMultiL2Event", false);
  descriptions.addWithDefaultLabel(desc);
}

bool HLTPixelIsolTrackL1TFilter::hltFilter(edm::Event& iEvent,
                                           const edm::EventSetup& iSetup,
                                           trigger::TriggerFilterObjectWithRefs& filterproduct) const {
  if (saveTags())
    filterproduct.addCollectionTag(candTag_);

  // Ref to Candidate object to be recorded in filter object
  edm::Ref<reco::IsolatedPixelTrackCandidateCollection> candref;

  // get hold of filtered candidates
  auto const recotrackcands = iEvent.getHandle(candToken_);

  //Filtering
  int n = 0;
  for (unsigned int i = 0; i < recotrackcands->size(); i++) {
    candref = edm::Ref<reco::IsolatedPixelTrackCandidateCollection>(recotrackcands, i);

    // select on transverse momentum
    if (!filterE_ && (candref->maxPtPxl() < maxptnearby_) && (candref->pt() > minpttrack_) &&
        fabs(candref->track()->eta()) < maxetatrack_ && fabs(candref->track()->eta()) > minetatrack_) {
      filterproduct.addObject(trigger::TriggerTrack, candref);
      n++;

      LogDebug("HcalIsoTrack") << "PixelIsolP:Candidate[" << n << "] pt|eta|phi " << candref->pt() << "|"
                               << candref->track()->pt() << "|" << candref->track()->eta() << "|"
                               << candref->track()->phi() << "\n";
    }

    // select on momentum
    if (filterE_) {
      if ((candref->maxPtPxl() < maxptnearby_) && ((candref->pt()) * cosh(candref->track()->eta()) > minEnergy_) &&
          fabs(candref->track()->eta()) < maxetatrack_ && fabs(candref->track()->eta()) > minetatrack_) {
        filterproduct.addObject(trigger::TriggerTrack, candref);
        n++;

        LogDebug("HcalIsoTrack") << "PixelIsolE:Candidate[" << n << "] pt|eta|phi " << candref->pt() << "|"
                                 << candref->track()->pt() << "|" << candref->track()->eta() << "|"
                                 << candref->track()->phi() << "\n";
      }
    }

    // stop looping over tracks if max number is reached
    if (!dropMultiL2Event_ && n >= nMaxTrackCandidates_)
      break;

  }  // loop over tracks

  bool accept(n > 0);

  if (dropMultiL2Event_ && n > nMaxTrackCandidates_)
    accept = false;

  LogDebug("HcalIsoTrack") << "PixelIsolL1Filter: Tracks " << n << " accept " << accept << "\n";

  return accept;
}

// declare this class as a framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTPixelIsolTrackL1TFilter);
