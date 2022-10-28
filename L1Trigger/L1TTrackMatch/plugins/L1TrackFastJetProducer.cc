///////////////////////////////////////////////////////////////////////////
//                                                                       //
// Producer of TkJet,                                                    //
// Cluster L1 tracks using fastjet                                       //
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

class L1TrackFastJetProducer : public edm::stream::EDProducer<> {
public:
  typedef TTTrack<Ref_Phase2TrackerDigi_> L1TTTrackType;
  typedef std::vector<L1TTTrackType> L1TTTrackCollectionType;

  explicit L1TrackFastJetProducer(const edm::ParameterSet&);
  ~L1TrackFastJetProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  //virtual void beginJob();
  void produce(edm::Event&, const edm::EventSetup&) override;
  //virtual void endJob();

  // track selection criteria
  const float trkZMax_;          // in [cm]
  const float trkChi2dofMax_;    // maximum track chi2dof
  const double trkBendChi2Max_;  // maximum track bendchi2
  const float trkPtMin_;         // in [GeV]
  const float trkEtaMax_;        // in [rad]
  const int trkNStubMin_;        // minimum number of stubs
  const int trkNPSStubMin_;      // minimum number of PS stubs
  const double deltaZ0Cut_;      // save with |L1z-z0| < maxZ0
  const double coneSize_;        // Use anti-kt with this cone size
  const bool doTightChi2_;
  const float trkPtTightChi2_;
  const float trkChi2dofTightChi2_;
  const bool displaced_;  //use prompt/displaced tracks

  const edm::EDGetTokenT<std::vector<TTTrack<Ref_Phase2TrackerDigi_> > > trackToken_;
  edm::EDGetTokenT<VertexCollection> pvToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;
};

// constructor
L1TrackFastJetProducer::L1TrackFastJetProducer(const edm::ParameterSet& iConfig)
    : trkZMax_((float)iConfig.getParameter<double>("trk_zMax")),
      trkChi2dofMax_((float)iConfig.getParameter<double>("trk_chi2dofMax")),
      trkBendChi2Max_(iConfig.getParameter<double>("trk_bendChi2Max")),
      trkPtMin_((float)iConfig.getParameter<double>("trk_ptMin")),
      trkEtaMax_((float)iConfig.getParameter<double>("trk_etaMax")),
      trkNStubMin_((int)iConfig.getParameter<int>("trk_nStubMin")),
      trkNPSStubMin_((int)iConfig.getParameter<int>("trk_nPSStubMin")),
      deltaZ0Cut_((float)iConfig.getParameter<double>("deltaZ0Cut")),
      coneSize_((float)iConfig.getParameter<double>("coneSize")),
      doTightChi2_(iConfig.getParameter<bool>("doTightChi2")),
      trkPtTightChi2_((float)iConfig.getParameter<double>("trk_ptTightChi2")),
      trkChi2dofTightChi2_((float)iConfig.getParameter<double>("trk_chi2dofTightChi2")),
      displaced_(iConfig.getParameter<bool>("displaced")),
      trackToken_(consumes<std::vector<TTTrack<Ref_Phase2TrackerDigi_> > >(
          iConfig.getParameter<edm::InputTag>("L1TrackInputTag"))),
      pvToken_(consumes<VertexCollection>(iConfig.getParameter<edm::InputTag>("L1PrimaryVertexTag"))),
      tTopoToken_(esConsumes<TrackerTopology, TrackerTopologyRcd>(edm::ESInputTag("", ""))) {
  if (displaced_)
    produces<TkJetCollection>("L1TrackFastJetsExtended");
  else
    produces<TkJetCollection>("L1TrackFastJets");
}

// destructor
L1TrackFastJetProducer::~L1TrackFastJetProducer() {}

// producer
void L1TrackFastJetProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  std::unique_ptr<TkJetCollection> L1TrackFastJets(new TkJetCollection);

  // L1 tracks
  edm::Handle<std::vector<TTTrack<Ref_Phase2TrackerDigi_> > > TTTrackHandle;
  iEvent.getByToken(trackToken_, TTTrackHandle);
  std::vector<TTTrack<Ref_Phase2TrackerDigi_> >::const_iterator iterL1Track;

  // Tracker Topology
  const TrackerTopology& tTopo = iSetup.getData(tTopoToken_);

  edm::Handle<l1t::VertexCollection> L1PrimaryVertexHandle;
  iEvent.getByToken(pvToken_, L1PrimaryVertexHandle);

  fastjet::JetDefinition jet_def(fastjet::antikt_algorithm, coneSize_);
  std::vector<fastjet::PseudoJet> JetInputs;

  float recoVtx = L1PrimaryVertexHandle->begin()->z0();
  unsigned int this_l1track = 0;
  for (iterL1Track = TTTrackHandle->begin(); iterL1Track != TTTrackHandle->end(); iterL1Track++) {
    this_l1track++;
    float trk_pt = iterL1Track->momentum().perp();
    float trk_z0 = iterL1Track->z0();
    float trk_chi2dof = iterL1Track->chi2Red();
    float trk_bendchi2 = iterL1Track->stubPtConsistency();
    std::vector<edm::Ref<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_> >, TTStub<Ref_Phase2TrackerDigi_> > >
        theStubs = iterL1Track->getStubRefs();
    int trk_nstub = (int)theStubs.size();

    if (std::abs(trk_z0) > trkZMax_)
      continue;
    if (std::abs(iterL1Track->momentum().eta()) > trkEtaMax_)
      continue;
    if (trk_pt < trkPtMin_)
      continue;
    if (trk_nstub < trkNStubMin_)
      continue;
    if (trk_chi2dof > trkChi2dofMax_)
      continue;
    if (trk_bendchi2 > trkBendChi2Max_)
      continue;
    if (doTightChi2_ && (trk_pt > trkPtTightChi2_ && trk_chi2dof > trkChi2dofTightChi2_))
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
    if (std::abs(recoVtx - trk_z0) > deltaZ0Cut_)
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
    iEvent.put(std::move(L1TrackFastJets), "L1TrackFastJetsExtended");
  else
    iEvent.put(std::move(L1TrackFastJets), "L1TrackFastJets");
}

//void L1TrackFastJetProducer::beginJob() {}

//void L1TrackFastJetProducer::endJob() {}

void L1TrackFastJetProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TrackFastJetProducer);
