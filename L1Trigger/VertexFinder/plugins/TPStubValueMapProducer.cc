// -*- C++ -*-
//
// Package:    L1Trigger/VertexFinder
// Class:      TPStubValueMapProducer
//
/**\class TPStubValueMapProducer TPStubValueMapProducer.cc L1Trigger/VertexFinder/plugins/TPStubValueMapProducer.cc

 Description: Produces an two value maps
  - a map which stores a TP object for every edm::Ptr<TrackingParticle>
  - a map which stores a Stub object for every TTStubRef

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Alexx Perloff
//         Created:  Mon, 08 Feb 2021 06:11:00 GMT
//
//

// system include files
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

// user include files
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/L1TrackTrigger/interface/TTStub.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "L1Trigger/VertexFinder/interface/AnalysisSettings.h"
#include "L1Trigger/VertexFinder/interface/TP.h"
#include "L1Trigger/VertexFinder/interface/Stub.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTClusterAssociationMap.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTStubAssociationMap.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTTrackAssociationMap.h"

//
// class declaration
//

class TPStubValueMapProducer : public edm::global::EDProducer<> {
public:
  explicit TPStubValueMapProducer(const edm::ParameterSet&);
  ~TPStubValueMapProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  // ----------constants, enums and typedefs ---------
  typedef edm::Ptr<TrackingParticle> TrackingParticlePtr;
  typedef edmNew::DetSet<TTStub<Ref_Phase2TrackerDigi_>> DetSet;
  typedef edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>> DetSetVec;
  typedef edm::Ref<DetSetVec, TTStub<Ref_Phase2TrackerDigi_>> TTStubRef;
  typedef TTStubAssociationMap<Ref_Phase2TrackerDigi_> TTStubAssMap;
  typedef TTClusterAssociationMap<Ref_Phase2TrackerDigi_> TTClusterAssMap;
  typedef TTTrackAssociationMap<Ref_Phase2TrackerDigi_> TTTrackAssMap;

  // ----------member data ---------------------------
  const std::vector<std::string> outputCollectionNames_;
  l1tVertexFinder::AnalysisSettings settings_;
  edm::EDGetTokenT<TTTrackAssociationMap<Ref_Phase2TrackerDigi_>> l1TracksMapToken_;
  const edm::EDGetTokenT<TrackingParticleCollection> tpToken_;
  const edm::EDGetTokenT<DetSetVec> stubToken_;
  const edm::EDGetTokenT<TTStubAssMap> stubTruthToken_;
  const edm::EDGetTokenT<TTClusterAssMap> clusterTruthToken_;

  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tGeomToken_;
};

//
// constructors and destructor
//
TPStubValueMapProducer::TPStubValueMapProducer(const edm::ParameterSet& iConfig)
    : outputCollectionNames_(iConfig.getParameter<std::vector<std::string>>("outputCollectionNames")),
      settings_(iConfig),
      l1TracksMapToken_(consumes<TTTrackAssMap>(iConfig.getParameter<edm::InputTag>("l1TracksTruthMapInputTags"))),
      tpToken_(consumes<TrackingParticleCollection>(iConfig.getParameter<edm::InputTag>("tpInputTag"))),
      stubToken_(consumes<DetSetVec>(iConfig.getParameter<edm::InputTag>("stubInputTag"))),
      stubTruthToken_(consumes<TTStubAssMap>(iConfig.getParameter<edm::InputTag>("stubTruthInputTag"))),
      clusterTruthToken_(consumes<TTClusterAssMap>(iConfig.getParameter<edm::InputTag>("clusterTruthInputTag"))),
      tTopoToken_(esConsumes<TrackerTopology, TrackerTopologyRcd>(edm::ESInputTag("", ""))),
      tGeomToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>(edm::ESInputTag("", ""))) {
  // Define EDM output to be written to file (if required)
  produces<TrackingParticleCollection>();
  produces<edm::ValueMap<l1tVertexFinder::TP>>(outputCollectionNames_[0]);
  produces<edm::ValueMap<l1tVertexFinder::TP>>(outputCollectionNames_[1]);
  produces<std::vector<l1tVertexFinder::TP>>(outputCollectionNames_[2]);
}

TPStubValueMapProducer::~TPStubValueMapProducer() {}

//
// member functions
//

// ------------ method called to produce the data  ------------
void TPStubValueMapProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  auto vTrackingParticles = std::make_unique<TrackingParticleCollection>();

  edm::Handle<TTTrackAssMap> mcTruthTTTrackHandle;
  edm::Handle<TrackingParticleCollection> tpHandle;
  edm::Handle<DetSetVec> ttStubHandle;
  edm::Handle<TTStubAssMap> mcTruthTTStubHandle;
  edm::Handle<TTClusterAssMap> mcTruthTTClusterHandle;
  iEvent.getByToken(l1TracksMapToken_, mcTruthTTTrackHandle);
  iEvent.getByToken(tpToken_, tpHandle);
  iEvent.getByToken(stubToken_, ttStubHandle);
  iEvent.getByToken(stubTruthToken_, mcTruthTTStubHandle);
  iEvent.getByToken(clusterTruthToken_, mcTruthTTClusterHandle);

  // Produce the vector of TP for the edm::Ref<TrackingParticle>->TP value map
  unsigned int nTP = tpHandle->size();
  auto vTPs = std::make_unique<std::vector<l1tVertexFinder::TP>>();
  auto vTPsUse = std::make_unique<std::vector<l1tVertexFinder::TP>>();
  vTPs->reserve(nTP);
  vTPsUse->reserve(nTP);
  std::set<edm::Ptr<TrackingParticle>> sTPs;
  for (unsigned int i = 0; i < nTP; i++) {
    TrackingParticlePtr tpPtr(tpHandle, i);
    // Store the TrackingParticle info, using class TP to provide easy access to the most useful info.
    l1tVertexFinder::TP tp(tpPtr, settings_);
    // Only bother storing tp if it could be useful for tracking efficiency or fake rate measurements.
    // Also create map relating edm::Ptr<TrackingParticle> to TP.
    if (tp.use()) {
      vTrackingParticles->push_back(tpHandle->at(i));
      vTPsUse->push_back(tp);
      sTPs.insert(tpPtr);
    }
    vTPs->push_back(tp);
  }

  auto vAllMatchedTPs = std::make_unique<std::vector<l1tVertexFinder::TP>>(*vTPsUse);
  for (auto& entry : mcTruthTTTrackHandle->getTTTrackToTrackingParticleMap()) {
    if (sTPs.count(entry.second) == 0) {
      vAllMatchedTPs->push_back(l1tVertexFinder::TP(entry.second, settings_));
    }
  }

  // Get the tracker geometry info needed to unpack the stub info.
  const TrackerTopology& tTopo = iSetup.getData(tTopoToken_);
  const TrackerGeometry& tGeom = iSetup.getData(tGeomToken_);

  const TrackerTopology* tTopology = &tTopo;
  const TrackerGeometry* tGeometry = &tGeom;

  //Create the vector of Stub for the TTStubRef->Stub value map
  unsigned int nStubs = ttStubHandle->size();
  auto vAllStubs = std::make_unique<std::vector<l1tVertexFinder::Stub>>();
  vAllStubs->reserve(nStubs);
  for (DetSetVec::const_iterator p_module = ttStubHandle->begin(); p_module != ttStubHandle->end(); p_module++) {
    for (DetSet::const_iterator p_ttstub = p_module->begin(); p_ttstub != p_module->end(); p_ttstub++) {
      TTStubRef ttStubRef = edmNew::makeRefTo(ttStubHandle, p_ttstub);
      // Store the Stub info, using class Stub to provide easy access to the most useful info.
      l1tVertexFinder::Stub stub(ttStubRef, settings_, tGeometry, tTopology);
      // Also fill truth associating stubs to tracking particles.
      stub.fillTruth(mcTruthTTStubHandle, mcTruthTTClusterHandle);
      vAllStubs->push_back(stub);
    }
  }

  //Set the Stubs associate to each TP
  std::map<const TrackingParticlePtr, std::vector<l1tVertexFinder::Stub>> tpStubMap;
  for (const l1tVertexFinder::TP& tp : *vTPs)
    tpStubMap[tp.getTrackingParticle()] = std::vector<l1tVertexFinder::Stub>();
  for (const l1tVertexFinder::Stub& stub : *vAllStubs) {
    for (const TrackingParticlePtr& tp : stub.assocTPs()) {
      tpStubMap[tp].push_back(stub);
    }
  }
  for (l1tVertexFinder::TP& tp : *vTPs) {
    assert(tpStubMap.count(tp.getTrackingParticle()) == 1);
    tp.setMatchingStubs(tpStubMap.find(tp.getTrackingParticle())->second);
  }

  //Put the products into the event
  // Modeled after: https://github.com/cms-sw/cmssw/blob/master/PhysicsTools/SelectorUtils/interface/VersionedIdProducer.h

  // Collections of products
  auto vTrackingParticlesHandle = iEvent.put(std::move(vTrackingParticles));
  auto vAllMatchedTPsHandle = iEvent.put(std::move(vAllMatchedTPs), outputCollectionNames_[2]);

  // Value maps to TP/Stub
  auto TPV = std::make_unique<edm::ValueMap<l1tVertexFinder::TP>>();
  edm::ValueMap<l1tVertexFinder::TP>::Filler fillerTP(*TPV);
  fillerTP.insert(tpHandle, vTPs->begin(), vTPs->end());
  fillerTP.fill();
  iEvent.put(std::move(TPV), outputCollectionNames_[0]);

  auto TPuseV = std::make_unique<edm::ValueMap<l1tVertexFinder::TP>>();
  edm::ValueMap<l1tVertexFinder::TP>::Filler fillerTPuse(*TPuseV);
  fillerTPuse.insert(vTrackingParticlesHandle, vTPsUse->begin(), vTPsUse->end());
  fillerTPuse.fill();
  iEvent.put(std::move(TPuseV), outputCollectionNames_[1]);
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void TPStubValueMapProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(TPStubValueMapProducer);
