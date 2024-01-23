///////////////////////////////////////////////////////////////////////////
//                                                                       //
// Producer of TPFastJets,                                               //
// Cluster tracking particles using fastjet                              //
//                                                                       //
// Created by: Claire Savard (Oct. 2023)                                 //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/Math/interface/LorentzVector.h"

// L1 objects
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TCorrelator/interface/TkJet.h"
#include "DataFormats/L1TCorrelator/interface/TkJetFwd.h"
#include "DataFormats/L1Trigger/interface/Vertex.h"
#include "DataFormats/L1TrackTrigger/interface/TTStub.h"

// truth object
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTStubAssociationMap.h"

// geometry
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include <fastjet/JetDefinition.hh>

#include <string>
#include "TMath.h"
#include "TH1.h"

using namespace l1t;
using namespace edm;
using namespace std;

//////////////////////////////
//                          //
//     CLASS DEFINITION     //
//                          //
//////////////////////////////

class TPFastJetProducer : public edm::stream::EDProducer<> {
public:
  explicit TPFastJetProducer(const edm::ParameterSet&);
  ~TPFastJetProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  // track selection criteria
  const float tpPtMin_;
  const float tpEtaMax_;
  const float tpZMax_;
  const int tpNStubMin_;
  const int tpNStubLayerMin_;
  const float coneSize_;  // Use anti-kt with this cone size

  edm::EDGetTokenT<std::vector<TrackingParticle>> trackingParticleToken_;
  edm::EDGetTokenT<TTStubAssociationMap<Ref_Phase2TrackerDigi_>> ttStubMCTruthToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;
};

// constructor
TPFastJetProducer::TPFastJetProducer(const edm::ParameterSet& iConfig)
    : tpPtMin_((float)iConfig.getParameter<double>("tp_ptMin")),
      tpEtaMax_((float)iConfig.getParameter<double>("tp_etaMax")),
      tpZMax_((float)iConfig.getParameter<double>("tp_zMax")),
      tpNStubMin_((int)iConfig.getParameter<int>("tp_nStubMin")),
      tpNStubLayerMin_((int)iConfig.getParameter<int>("tp_nStubLayerMin")),
      coneSize_((float)iConfig.getParameter<double>("coneSize")),
      trackingParticleToken_(
          consumes<std::vector<TrackingParticle>>(iConfig.getParameter<edm::InputTag>("TrackingParticleInputTag"))),
      ttStubMCTruthToken_(consumes<TTStubAssociationMap<Ref_Phase2TrackerDigi_>>(
          iConfig.getParameter<edm::InputTag>("MCTruthStubInputTag"))),
      tTopoToken_(esConsumes<TrackerTopology, TrackerTopologyRcd>(edm::ESInputTag("", ""))) {
  produces<TkJetCollection>("TPFastJets");
}

// producer
void TPFastJetProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  std::unique_ptr<TkJetCollection> TPFastJets(new TkJetCollection);

  // Tracking particles
  edm::Handle<std::vector<TrackingParticle>> TrackingParticleHandle;
  iEvent.getByToken(trackingParticleToken_, TrackingParticleHandle);
  std::vector<TrackingParticle>::const_iterator iterTP;

  // MC truth association maps
  edm::Handle<TTStubAssociationMap<Ref_Phase2TrackerDigi_>> MCTruthTTStubHandle;
  iEvent.getByToken(ttStubMCTruthToken_, MCTruthTTStubHandle);

  // Tracker Topology
  const TrackerTopology& tTopo = iSetup.getData(tTopoToken_);

  fastjet::JetDefinition jet_def(fastjet::antikt_algorithm, coneSize_);
  std::vector<fastjet::PseudoJet> JetInputs;

  // loop over tps
  unsigned int this_tp = 0;
  for (iterTP = TrackingParticleHandle->begin(); iterTP != TrackingParticleHandle->end(); iterTP++) {
    edm::Ptr<TrackingParticle> tp_ptr(TrackingParticleHandle, this_tp);
    this_tp++;

    std::vector<edm::Ref<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>, TTStub<Ref_Phase2TrackerDigi_>>>
        theStubRefs = MCTruthTTStubHandle->findTTStubRefs(tp_ptr);
    int nStubTP = (int)theStubRefs.size();

    // how many layers/disks have stubs?
    int hasStubInLayer[11] = {0};
    for (auto& theStubRef : theStubRefs) {
      DetId detid(theStubRef->getDetId());

      int layer = -1;
      if (detid.subdetId() == StripSubdetector::TOB) {
        layer = static_cast<int>(tTopo.layer(detid)) - 1;  //fill in array as entries 0-5
      } else if (detid.subdetId() == StripSubdetector::TID) {
        layer = static_cast<int>(tTopo.layer(detid)) + 5;  //fill in array as entries 6-10
      }

      //treat genuine stubs separately (==2 is genuine, ==1 is not)
      if (MCTruthTTStubHandle->findTrackingParticlePtr(theStubRef).isNull() && hasStubInLayer[layer] < 2)
        hasStubInLayer[layer] = 1;
      else
        hasStubInLayer[layer] = 2;
    }

    int nStubLayerTP = 0;
    for (int isum : hasStubInLayer) {
      if (isum >= 1)
        nStubLayerTP += 1;
    }

    // tp quality cuts to match L1 tracks
    if (iterTP->pt() < tpPtMin_)
      continue;
    if (fabs(iterTP->eta()) > tpEtaMax_)
      continue;
    if (nStubTP < tpNStubMin_)
      continue;
    if (nStubLayerTP < tpNStubLayerMin_)
      continue;
    if (fabs(iterTP->z0()) > tpZMax_)
      continue;
    if (iterTP->charge() == 0.)  // extra check that all tps are charged
      continue;
    if (iterTP->eventId().event() > 0)  // only select hard interaction tps
      continue;

    fastjet::PseudoJet psuedoJet(iterTP->px(), iterTP->py(), iterTP->pz(), iterTP->energy());
    JetInputs.push_back(psuedoJet);                // input tps for clustering
    JetInputs.back().set_user_index(this_tp - 1);  // save tp index in the collection
  }                                                // end loop over tps

  fastjet::ClusterSequence cs(JetInputs, jet_def);  // define the output jet collection
  std::vector<fastjet::PseudoJet> JetOutputs =
      fastjet::sorted_by_pt(cs.inclusive_jets(0));  // output jet collection, pT-ordered

  for (unsigned int ijet = 0; ijet < JetOutputs.size(); ++ijet) {
    math::XYZTLorentzVector jetP4(
        JetOutputs[ijet].px(), JetOutputs[ijet].py(), JetOutputs[ijet].pz(), JetOutputs[ijet].modp());
    float sumpt = 0;
    float avgZ = 0;
    std::vector<edm::Ptr<TrackingParticle>> tpPtrs;
    std::vector<fastjet::PseudoJet> fjConstituents = fastjet::sorted_by_pt(cs.constituents(JetOutputs[ijet]));

    for (unsigned int i = 0; i < fjConstituents.size(); ++i) {
      auto index = fjConstituents[i].user_index();
      edm::Ptr<TrackingParticle> tpPtr(TrackingParticleHandle, index);
      tpPtrs.push_back(tpPtr);  // tracking particles in the jet
      sumpt = sumpt + tpPtr->pt();
      avgZ = avgZ + tpPtr->pt() * tpPtr->z0();
    }
    avgZ = avgZ / sumpt;
    edm::Ref<JetBxCollection> jetRef;
    std::vector<edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_>>> dummyL1TrackPtrs;  // can't create TkJet with tp references
    TkJet tpJet(jetP4, dummyL1TrackPtrs, avgZ, fjConstituents.size(), 0, 0, 0, false);
    TPFastJets->push_back(tpJet);
  }  //end loop over Jet Outputs

  iEvent.put(std::move(TPFastJets), "TPFastJets");
}

void TPFastJetProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // The following says we do not know what parameters are allowed so do no validation
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("TrackingParticleInputTag", edm::InputTag("mix", "MergedTrackTruth"));
  desc.add<edm::InputTag>("MCTruthStubInputTag", edm::InputTag("TTStubAssociatorFromPixelDigis", "StubAccepted"));
  desc.add<double>("tp_ptMin", 2.0);
  desc.add<double>("tp_etaMax", 2.4);
  desc.add<double>("tp_zMax", 15.);
  desc.add<int>("tp_nStubMin", 4);
  desc.add<int>("tp_nStubLayerMin", 4);
  desc.add<double>("coneSize", 0.4);
  descriptions.add("tpFastJets", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(TPFastJetProducer);
