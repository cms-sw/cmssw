#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"

// #include "Geometry/Records/interface/StackedTrackerGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

// TTStubAssociationMap.h forgets to two needed files, so must include them here ...
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTClusterAssociationMap.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTStubAssociationMap.h"

#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/GenJet.h"

#include "L1Trigger/TrackFindingTMTT/interface/InputData.h"
#include "L1Trigger/TrackFindingTMTT/interface/Settings.h"

#include <map>

using namespace std;

namespace TMTT {
 
InputData::InputData(const edm::Event& iEvent, const edm::EventSetup& iSetup, Settings* settings, 
  const edm::EDGetTokenT<TrackingParticleCollection> tpInputTag,
  const edm::EDGetTokenT<DetSetVec> stubInputTag,
  const edm::EDGetTokenT<TTStubAssMap> stubTruthInputTag,
  const edm::EDGetTokenT<TTClusterAssMap> clusterTruthInputTag,
  const edm::EDGetTokenT< reco::GenJetCollection > genJetInputTag
   )
  {

  vTPs_.reserve(2500);
  vStubs_.reserve(35000);
  vAllStubs_.reserve(35000);

  // Note if job will use MC truth info (or skip it to save CPU).
  enableMCtruth_ = settings->enableMCtruth();

  // Get TrackingParticle info

  edm::Handle<TrackingParticleCollection> tpHandle;

  if (enableMCtruth_) {
    iEvent.getByToken(tpInputTag, tpHandle );

    unsigned int tpCount = 0;
    for (unsigned int i = 0; i < tpHandle->size(); i++) {
      const TrackingParticle& tPart = tpHandle->at(i);
      // Creating Ptr uses CPU, so apply Pt cut here, copied from TP::fillUse(), to avoid doing it too often. 
      const float ptMin = min(settings->genMinPt(), 0.7*settings->houghMinPt());
      if (tPart.pt() > ptMin) {
	TrackingParticlePtr tpPtr(tpHandle, i);
	// Store the TrackingParticle info, using class TP to provide easy access to the most useful info.
	TP tp(tpPtr, tpCount, settings);
	// Only bother storing tp if it could be useful for tracking efficiency or fake rate measurements.
	if (tp.use()) {

    edm::Handle< reco::GenJetCollection > genJetHandle;
    iEvent.getByToken(genJetInputTag, genJetHandle);
    if ( genJetHandle.isValid() ) {
      tp.fillNearestJetInfo( genJetHandle.product() );
    }

	  vTPs_.push_back( tp );
	  tpCount++;
	}
      }
    }
  }

  // Also create map relating edm::Ptr<TrackingParticle> to TP.

  map<edm::Ptr< TrackingParticle >, const TP* > translateTP;

  if (enableMCtruth_) {
    for (const TP& tp : vTPs_) {
      TrackingParticlePtr tpPtr(tp);
      translateTP[tpPtr] = &tp;
    }
  }

  // Get the tracker geometry info needed to unpack the stub info.

  edm::ESHandle<TrackerGeometry> trackerGeometryHandle;
  iSetup.get<TrackerDigiGeometryRecord>().get( trackerGeometryHandle );

  const TrackerGeometry*  trackerGeometry = trackerGeometryHandle.product();

  edm::ESHandle<TrackerTopology> trackerTopologyHandle;
  iSetup.get<TrackerTopologyRcd>().get(trackerTopologyHandle);

  const TrackerTopology*  trackerTopology = trackerTopologyHandle.product();

  // Get stub info, by looping over modules and then stubs inside each module.
  // Also get the association map from stubs to tracking particles.

  edm::Handle<DetSetVec>       ttStubHandle;
  edm::Handle<TTStubAssMap>    mcTruthTTStubHandle;
  edm::Handle<TTClusterAssMap> mcTruthTTClusterHandle;
  iEvent.getByToken(stubInputTag, ttStubHandle );
  if (enableMCtruth_) {
    iEvent.getByToken(stubTruthInputTag, mcTruthTTStubHandle );
    iEvent.getByToken(clusterTruthInputTag, mcTruthTTClusterHandle );
  }

  unsigned int stubCount = 0;

  for (DetSetVec::const_iterator p_module = ttStubHandle->begin(); p_module != ttStubHandle->end(); p_module++) {
    for (DetSet::const_iterator p_ttstub = p_module->begin(); p_ttstub != p_module->end(); p_ttstub++) {
      TTStubRef ttStubRef = edmNew::makeRefTo(ttStubHandle, p_ttstub );

      // Store the Stub info, using class Stub to provide easy access to the most useful info.
      Stub stub(ttStubRef, stubCount, settings, trackerGeometry, trackerTopology );
      // Also fill truth associating stubs to tracking particles.
      if (enableMCtruth_) stub.fillTruth(translateTP, mcTruthTTStubHandle, mcTruthTTClusterHandle); 
      vAllStubs_.push_back( stub );
      stubCount++;
    }
  }

  // Produced reduced list containing only the subset of stubs that the user has declared will be 
  // output by the front-end readout electronics.
  for (const Stub& s : vAllStubs_) {
    if (s.frontendPass()) vStubs_.push_back( &s );
  }
  // Optionally sort stubs according to bend, so highest Pt ones are sent from DTC to GP first.
  if (settings->orderStubsByBend()) std::sort(vStubs_.begin(), vStubs_.end(), SortStubsInBend());

  // Note list of stubs produced by each tracking particle.
  // (By passing vAllStubs_ here instead of vStubs_, it means that any algorithmic efficiencies
  // measured will be reduced if the tightened frontend electronics cuts, specified in section StubCuts
  // of Analyze_Defaults_cfi.py, are not 100% efficient).
  if (enableMCtruth_) {
    for (unsigned int j = 0; j < vTPs_.size(); j++) {
      vTPs_[j].fillTruth(vAllStubs_);
    }
  }
}

}
