#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/GenJet.h"

#include "L1Trigger/TrackFindingTMTT/interface/InputData.h"
#include "L1Trigger/TrackFindingTMTT/interface/Settings.h"
#include "L1Trigger/TrackFindingTMTT/interface/StubKiller.h"
#include "L1Trigger/TrackFindingTMTT/interface/StubWindowSuggest.h"
#include "L1Trigger/TrackFindingTMTT/interface/DegradeBend.h"

#include <map>
#include <memory>

using namespace std;

namespace tmtt {

  InputData::InputData(const edm::Event& iEvent,
                       const edm::EventSetup& iSetup,
                       const Settings* settings,
                       StubWindowSuggest* stubWindowSuggest,
                       const DegradeBend* degradeBend,
                       const TrackerGeometry* trackerGeometry,
                       const TrackerTopology* trackerTopology,
                       const list<TrackerModule>& listTrackerModule,
                       const edm::EDGetTokenT<TrackingParticleCollection> tpToken,
                       const edm::EDGetTokenT<TTStubDetSetVec> stubToken,
                       const edm::EDGetTokenT<TTStubAssMap> stubTruthToken,
                       const edm::EDGetTokenT<TTClusterAssMap> clusterTruthToken,
                       const edm::EDGetTokenT<reco::GenJetCollection> genJetToken)
      :  // Note if job will use MC truth info (or skip it to save CPU).
        enableMCtruth_(settings->enableMCtruth()) {
    edm::Handle<TrackingParticleCollection> tpHandle;
    edm::Handle<TTStubDetSetVec> ttStubHandle;
    edm::Handle<TTStubAssMap> mcTruthTTStubHandle;
    edm::Handle<TTClusterAssMap> mcTruthTTClusterHandle;
    edm::Handle<reco::GenJetCollection> genJetHandle;
    iEvent.getByToken(stubToken, ttStubHandle);
    if (enableMCtruth_) {
      iEvent.getByToken(tpToken, tpHandle);
      iEvent.getByToken(stubTruthToken, mcTruthTTStubHandle);
      iEvent.getByToken(clusterTruthToken, mcTruthTTClusterHandle);
      iEvent.getByToken(genJetToken, genJetHandle);
    }

    // Get TrackingParticle info

    if (enableMCtruth_) {
      unsigned int tpCount = 0;
      for (unsigned int i = 0; i < tpHandle->size(); i++) {
        const TrackingParticle& tPart = tpHandle->at(i);
        // Creating Ptr uses CPU, so apply Pt cut here, copied from TP::fillUse(), to avoid doing it too often.
        constexpr float ptMinScale = 0.7;
        const float ptMin = min(settings->genMinPt(), ptMinScale * settings->houghMinPt());
        if (tPart.pt() > ptMin) {
          TrackingParticlePtr tpPtr(tpHandle, i);
          // Store the TrackingParticle info, using class TP to provide easy access to the most useful info.
          TP tp(tpPtr, tpCount, settings);
          // Only bother storing tp if it could be useful for tracking efficiency or fake rate measurements.
          if (tp.use()) {
            if (genJetHandle.isValid()) {
              tp.fillNearestJetInfo(genJetHandle.product());
            }

            vTPs_.push_back(tp);
            tpCount++;
          }
        }
      }
    }

    // Also create map relating edm::Ptr<TrackingParticle> to TP.

    map<edm::Ptr<TrackingParticle>, const TP*> translateTP;

    if (enableMCtruth_) {
      for (const TP& tp : vTPs_) {
        const TrackingParticlePtr& tpPtr = tp.trackingParticlePtr();
        translateTP[tpPtr] = &tp;
      }
    }

    // Initialize code for killing some stubs to model detector problems.
    const StubKiller::KillOptions killOpt = static_cast<StubKiller::KillOptions>(settings->killScenario());
    std::unique_ptr<const StubKiller> stubKiller;
    if (killOpt != StubKiller::KillOptions::none) {
      stubKiller = std::make_unique<StubKiller>(killOpt, trackerTopology, trackerGeometry, iEvent);
    }

    // Loop over tracker modules to get module info & stubs.

    for (const TrackerModule& trackerModule : listTrackerModule) {
      const DetId& stackedDetId = trackerModule.stackedDetId();
      TTStubDetSetVec::const_iterator p_module = ttStubHandle->find(stackedDetId);
      if (p_module != ttStubHandle->end()) {
        for (TTStubDetSet::const_iterator p_ttstub = p_module->begin(); p_ttstub != p_module->end(); p_ttstub++) {
          TTStubRef ttStubRef = edmNew::makeRefTo(ttStubHandle, p_ttstub);
          const unsigned int stubIndex = vAllStubs_.size();

          // Store the Stub info, using class Stub to provide easy access to the most useful info.
          vAllStubs_.emplace_back(
              ttStubRef, stubIndex, settings, trackerTopology, &trackerModule, degradeBend, stubKiller.get());

          // Also fill truth associating stubs to tracking particles.
          if (enableMCtruth_) {
            Stub& stub = vAllStubs_.back();
            stub.fillTruth(translateTP, mcTruthTTStubHandle, mcTruthTTClusterHandle);
          }
        }
      }
    }

    // Produced reduced list containing only the subset of stubs that the user has declared will be
    // output by the front-end readout electronics.
    for (Stub& s : vAllStubs_) {
      if (s.frontendPass()) {
        vStubs_.push_back(&s);
        vStubsConst_.push_back(&s);
      }
    }
    // Optionally sort stubs according to bend, so highest Pt ones are sent from DTC to GP first.
    if (settings->orderStubsByBend()) {
      auto orderStubsByBend = [](const Stub* a, const Stub* b) { return (std::abs(a->bend()) < std::abs(b->bend())); };
      vStubs_.sort(orderStubsByBend);
    }

    // Note list of stubs produced by each tracking particle.
    // (By passing vAllStubs_ here instead of vStubs_, it means that any algorithmic efficiencies
    // measured will be reduced if the tightened frontend electronics cuts, specified in section StubCuts
    // of Analyze_Defaults_cfi.py, are not 100% efficient).
    if (enableMCtruth_) {
      for (TP& tp : vTPs_) {
        tp.fillTruth(vAllStubs_);
      }
    }

    // If requested, recommend better FE stub window cuts.
    if (settings->printStubWindows()) {
      for (const Stub& s : vAllStubs_) {
        stubWindowSuggest->process(trackerTopology, &s);
      }
    }
  }

}  // namespace tmtt
