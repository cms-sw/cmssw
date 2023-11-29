///////////////////////////////////////////////////////////////////////////
//                                                                       //
// Producer of TruTrkFastJet,                                            //
// Cluster L1 tracks with truth info using fastjet                       //
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

// MC
#include "SimTracker/TrackTriggerAssociation/interface/TTTrackAssociationMap.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

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

class L1TruthTrackFastJetProducer : public edm::stream::EDProducer<> {
public:
  typedef TTTrack<Ref_Phase2TrackerDigi_> L1TTTrackType;
  typedef std::vector<L1TTTrackType> L1TTTrackCollectionType;

  explicit L1TruthTrackFastJetProducer(const edm::ParameterSet&);
  ~L1TruthTrackFastJetProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  // track selection criteria
  const float trkZMax_;      // in [cm]
  const float trkPtMin_;     // in [GeV]
  const float trkEtaMax_;    // in [rad]
  const int trkNStubMin_;    // minimum number of stubs
  const int trkNPSStubMin_;  // minimum number of PS stubs
  const double coneSize_;    // Use anti-kt with this cone size
  const bool displaced_;     //use prompt/displaced tracks

  const edm::EDGetTokenT<std::vector<TTTrack<Ref_Phase2TrackerDigi_> > > trackToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;
  const edm::EDGetTokenT<TTTrackAssociationMap<Ref_Phase2TrackerDigi_> > ttTrackMCTruthToken_;
};

// constructor
L1TruthTrackFastJetProducer::L1TruthTrackFastJetProducer(const edm::ParameterSet& iConfig)
    : trkZMax_((float)iConfig.getParameter<double>("trk_zMax")),
      trkPtMin_((float)iConfig.getParameter<double>("trk_ptMin")),
      trkEtaMax_((float)iConfig.getParameter<double>("trk_etaMax")),
      trkNStubMin_((int)iConfig.getParameter<int>("trk_nStubMin")),
      trkNPSStubMin_((int)iConfig.getParameter<int>("trk_nPSStubMin")),
      coneSize_((float)iConfig.getParameter<double>("coneSize")),
      displaced_(iConfig.getParameter<bool>("displaced")),
      trackToken_(consumes<std::vector<TTTrack<Ref_Phase2TrackerDigi_> > >(
          iConfig.getParameter<edm::InputTag>("L1TrackInputTag"))),
      tTopoToken_(esConsumes<TrackerTopology, TrackerTopologyRcd>(edm::ESInputTag("", ""))),
      ttTrackMCTruthToken_(consumes<TTTrackAssociationMap<Ref_Phase2TrackerDigi_> >(
          iConfig.getParameter<edm::InputTag>("MCTruthTrackInputTag"))) {
  if (displaced_)
    produces<TkJetCollection>("L1TruthTrackFastJetsExtended");
  else
    produces<TkJetCollection>("L1TruthTrackFastJets");
}

// producer
void L1TruthTrackFastJetProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  std::unique_ptr<TkJetCollection> L1TrackFastJets(new TkJetCollection);

  // L1 tracks
  edm::Handle<std::vector<TTTrack<Ref_Phase2TrackerDigi_> > > TTTrackHandle;
  iEvent.getByToken(trackToken_, TTTrackHandle);
  std::vector<TTTrack<Ref_Phase2TrackerDigi_> >::const_iterator iterL1Track;

  // MC truth
  edm::Handle<TTTrackAssociationMap<Ref_Phase2TrackerDigi_> > MCTruthTTTrackHandle;
  iEvent.getByToken(ttTrackMCTruthToken_, MCTruthTTTrackHandle);

  // Tracker Topology
  const TrackerTopology& tTopo = iSetup.getData(tTopoToken_);

  fastjet::JetDefinition jet_def(fastjet::antikt_algorithm, coneSize_);
  std::vector<fastjet::PseudoJet> JetInputs;

  unsigned int this_l1track = 0;
  for (iterL1Track = TTTrackHandle->begin(); iterL1Track != TTTrackHandle->end(); iterL1Track++) {
    edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_> > l1track_ptr(TTTrackHandle, this_l1track);
    this_l1track++;
    std::vector<edm::Ref<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_> >, TTStub<Ref_Phase2TrackerDigi_> > >
        theStubs = iterL1Track->getStubRefs();

    // standard quality cuts
    if (std::abs(iterL1Track->z0()) > trkZMax_)
      continue;
    if (std::abs(iterL1Track->momentum().eta()) > trkEtaMax_)
      continue;
    if (iterL1Track->momentum().perp() < trkPtMin_)
      continue;
    int trk_nstub = (int)theStubs.size();
    if (trk_nstub < trkNStubMin_)
      continue;

    int trk_nPS = 0;
    for (int istub = 0; istub < trk_nstub; istub++) {
      DetId detId(theStubs.at(istub)->getDetId());
      bool tmp_isPS = false;
      if (detId.det() == DetId::Detector::Tracker) {
        if (detId.subdetId() == StripSubdetector::TOB && tTopo.tobLayer(detId) <= 3)
          tmp_isPS = true;
        else if (detId.subdetId() == StripSubdetector::TID && tTopo.tidRing(detId) <= 9)
          tmp_isPS = true;
      }
      if (tmp_isPS)
        trk_nPS++;
    }
    if (trk_nPS < trkNPSStubMin_)
      continue;

    // check that trk is real and from hard interaction
    edm::Ptr<TrackingParticle> my_tp = MCTruthTTTrackHandle->findTrackingParticlePtr(l1track_ptr);
    if (my_tp.isNull())  // there is no tp match so the track is fake
      continue;
    int tp_eventid = my_tp->eventId().event();
    if (tp_eventid > 0)  // matched tp is from pileup
      continue;

    fastjet::PseudoJet psuedoJet(iterL1Track->momentum().x(),
                                 iterL1Track->momentum().y(),
                                 iterL1Track->momentum().z(),
                                 iterL1Track->momentum().mag());
    JetInputs.push_back(psuedoJet);                     // input tracks for clustering
    JetInputs.back().set_user_index(this_l1track - 1);  // save track index in the collection
  }                                                     // end loop over tracks

  fastjet::ClusterSequence cs(JetInputs, jet_def);  // define the output jet collection
  std::vector<fastjet::PseudoJet> JetOutputs =
      fastjet::sorted_by_pt(cs.inclusive_jets(0));  // output jet collection, pT-ordered

  for (unsigned int ijet = 0; ijet < JetOutputs.size(); ++ijet) {
    math::XYZTLorentzVector jetP4(
        JetOutputs[ijet].px(), JetOutputs[ijet].py(), JetOutputs[ijet].pz(), JetOutputs[ijet].modp());
    float sumpt = 0;
    float avgZ = 0;
    std::vector<edm::Ptr<L1TTTrackType> > L1TrackPtrs;
    std::vector<fastjet::PseudoJet> fjConstituents = fastjet::sorted_by_pt(cs.constituents(JetOutputs[ijet]));

    for (unsigned int i = 0; i < fjConstituents.size(); ++i) {
      auto index = fjConstituents[i].user_index();
      edm::Ptr<L1TTTrackType> trkPtr(TTTrackHandle, index);
      L1TrackPtrs.push_back(trkPtr);  // L1Tracks in the jet
      sumpt = sumpt + trkPtr->momentum().perp();
      avgZ = avgZ + trkPtr->momentum().perp() * trkPtr->z0();
    }
    avgZ = avgZ / sumpt;
    edm::Ref<JetBxCollection> jetRef;
    TkJet trkJet(jetP4, jetRef, L1TrackPtrs, avgZ);
    L1TrackFastJets->push_back(trkJet);
  }  //end loop over Jet Outputs

  if (displaced_)
    iEvent.put(std::move(L1TrackFastJets), "L1TruthTrackFastJetsExtended");
  else
    iEvent.put(std::move(L1TrackFastJets), "L1TruthTrackFastJets");
}

void L1TruthTrackFastJetProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // The following says we do not know what parameters are allowed so do no validation
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("L1TrackInputTag", edm::InputTag("l1tTTTracksFromTrackletEmulation", "Level1TTTracks"));
  desc.add<edm::InputTag>("MCTruthTrackInputTag", edm::InputTag("TTTrackAssociatorFromPixelDigis", "Level1TTTracks"));
  desc.add<double>("trk_zMax", 15.);
  desc.add<double>("trk_ptMin", 2.0);
  desc.add<double>("trk_etaMax", 2.4);
  desc.add<int>("trk_nStubMin", 4);
  desc.add<int>("trk_nPSStubMin", -1);
  desc.add<double>("coneSize", 0.4);
  desc.add<bool>("displaced", false);
  descriptions.add("l1tTruthTrackFastJets", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TruthTrackFastJetProducer);
