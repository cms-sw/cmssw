/* class PFRecoTauTagInfoProducer
 * returns a PFTauTagInfo collection starting from a JetTrackAssociations <a PFJet,a list of Tracks> collection,
 * created: Aug 28 2007,
 * revised: ,
 * authors: Ludovic Houchu
 */

#include "DataFormats/JetReco/interface/JetTracksAssociation.h"
#include "DataFormats/TauReco/interface/PFTauTagInfo.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "RecoTauTag/RecoTau/interface/PFRecoTauTagInfoAlgorithm.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <FWCore/ParameterSet/interface/ConfigurationDescriptions.h>
#include <FWCore/ParameterSet/interface/ParameterSetDescription.h>

#include "CLHEP/Random/RandGauss.h"

#include "Math/GenVector/VectorUtil.h"

#include <memory>

using namespace reco;
using namespace edm;
using namespace std;

class PFRecoTauTagInfoProducer : public edm::global::EDProducer<> {
public:
  explicit PFRecoTauTagInfoProducer(const edm::ParameterSet& iConfig);
  ~PFRecoTauTagInfoProducer() override;
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  std::unique_ptr<const PFRecoTauTagInfoAlgorithm> PFRecoTauTagInfoAlgo_;
  edm::InputTag PFCandidateProducer_;
  edm::InputTag PFJetTracksAssociatorProducer_;
  edm::InputTag PVProducer_;
  double smearedPVsigmaX_;
  double smearedPVsigmaY_;
  double smearedPVsigmaZ_;

  edm::EDGetTokenT<PFCandidateCollection> PFCandidate_token;
  edm::EDGetTokenT<JetTracksAssociationCollection> PFJetTracksAssociator_token;
  edm::EDGetTokenT<VertexCollection> PV_token;
};

PFRecoTauTagInfoProducer::PFRecoTauTagInfoProducer(const edm::ParameterSet& iConfig) {
  PFCandidateProducer_ = iConfig.getParameter<edm::InputTag>("PFCandidateProducer");
  PFJetTracksAssociatorProducer_ = iConfig.getParameter<edm::InputTag>("PFJetTracksAssociatorProducer");
  PVProducer_ = iConfig.getParameter<edm::InputTag>("PVProducer");
  smearedPVsigmaX_ = iConfig.getParameter<double>("smearedPVsigmaX");
  smearedPVsigmaY_ = iConfig.getParameter<double>("smearedPVsigmaY");
  smearedPVsigmaZ_ = iConfig.getParameter<double>("smearedPVsigmaZ");
  PFRecoTauTagInfoAlgo_ = std::make_unique<PFRecoTauTagInfoAlgorithm>(iConfig);
  PFCandidate_token = consumes<PFCandidateCollection>(PFCandidateProducer_);
  PFJetTracksAssociator_token = consumes<JetTracksAssociationCollection>(PFJetTracksAssociatorProducer_);
  PV_token = consumes<VertexCollection>(PVProducer_);
  produces<PFTauTagInfoCollection>();
}
PFRecoTauTagInfoProducer::~PFRecoTauTagInfoProducer() {}

void PFRecoTauTagInfoProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  edm::Handle<JetTracksAssociationCollection> thePFJetTracksAssociatorCollection;
  iEvent.getByToken(PFJetTracksAssociator_token, thePFJetTracksAssociatorCollection);
  // *** access the PFCandidateCollection in the event in order to retrieve the PFCandidateRefVector which constitutes each PFJet
  edm::Handle<PFCandidateCollection> thePFCandidateCollection;
  iEvent.getByToken(PFCandidate_token, thePFCandidateCollection);
  vector<CandidatePtr> thePFCandsInTheEvent;
  for (unsigned int i_PFCand = 0; i_PFCand != thePFCandidateCollection->size(); i_PFCand++) {
    thePFCandsInTheEvent.push_back(CandidatePtr(thePFCandidateCollection, i_PFCand));
  }
  // ***
  // query a rec/sim PV
  edm::Handle<VertexCollection> thePVs;
  iEvent.getByToken(PV_token, thePVs);
  const VertexCollection vertCollection = *(thePVs.product());
  math::XYZPoint V(0, 0, -1000.);

  Vertex thePV;
  if (!vertCollection.empty())
    thePV = *(vertCollection.begin());
  else {
    Vertex::Error SimPVError;
    SimPVError(0, 0) = 15. * 15.;
    SimPVError(1, 1) = 15. * 15.;
    SimPVError(2, 2) = 15. * 15.;
    Vertex::Point SimPVPoint(0., 0., -1000.);
    thePV = Vertex(SimPVPoint, SimPVError, 1, 1, 1);
  }

  auto resultExt = std::make_unique<PFTauTagInfoCollection>();
  for (JetTracksAssociationCollection::const_iterator iAssoc = thePFJetTracksAssociatorCollection->begin();
       iAssoc != thePFJetTracksAssociatorCollection->end();
       iAssoc++) {
    PFTauTagInfo myPFTauTagInfo = PFRecoTauTagInfoAlgo_->buildPFTauTagInfo(
        JetBaseRef((*iAssoc).first), thePFCandsInTheEvent, (*iAssoc).second, thePV);
    resultExt->push_back(myPFTauTagInfo);
  }

  //  OrphanHandle<PFTauTagInfoCollection> myPFTauTagInfoCollection=iEvent.put(std::move(resultExt));
  iEvent.put(std::move(resultExt));
}

void PFRecoTauTagInfoProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  {
    // pfRecoTauTagInfoProducerInsideOut
    edm::ParameterSetDescription desc;
    desc.add<int>("tkminTrackerHitsn", 3);
    desc.add<double>("tkminPt", 0.5);
    desc.add<double>("tkmaxChi2", 100.0);
    desc.add<double>("ChargedHadrCand_AssociationCone", 1.0);
    desc.add<int>("ChargedHadrCand_tkminTrackerHitsn", 3);
    desc.add<double>("ChargedHadrCand_tkmaxChi2", 100.0);
    desc.add<double>("tkPVmaxDZ", 0.2);
    desc.add<double>("GammaCand_EcalclusMinEt", 1.0);
    desc.add<int>("tkminPixelHitsn", 0);
    desc.add<edm::InputTag>("PVProducer", edm::InputTag("offlinePrimaryVertices"));
    desc.add<edm::InputTag>("PFCandidateProducer", edm::InputTag("particleFlow"));
    desc.add<double>("ChargedHadrCand_tkminPt", 0.5);
    desc.add<double>("ChargedHadrCand_tkmaxipt", 0.03);
    desc.add<int>("ChargedHadrCand_tkminPixelHitsn", 0);
    desc.add<bool>("UsePVconstraint", true);
    desc.add<double>("NeutrHadrCand_HcalclusMinEt", 1.0);
    desc.add<edm::InputTag>("PFJetTracksAssociatorProducer", edm::InputTag("insideOutJetTracksAssociatorAtVertex"));
    desc.add<double>("smearedPVsigmaY", 0.0015);
    desc.add<double>("smearedPVsigmaX", 0.0015);
    desc.add<double>("smearedPVsigmaZ", 0.005);
    desc.add<double>("ChargedHadrCand_tkPVmaxDZ", 0.2);
    desc.add<double>("tkmaxipt", 0.03);
    descriptions.add("pfRecoTauTagInfoProducerInsideOut", desc);
  }
  {
    // pfRecoTauTagInfoProducer
    edm::ParameterSetDescription desc;
    desc.add<int>("tkminTrackerHitsn", 3);
    desc.add<double>("tkminPt", 0.5);
    desc.add<double>("tkmaxChi2", 100.0);
    desc.add<double>("ChargedHadrCand_AssociationCone", 0.8);
    desc.add<int>("ChargedHadrCand_tkminTrackerHitsn", 3);
    desc.add<double>("ChargedHadrCand_tkmaxChi2", 100.0);
    desc.add<double>("tkPVmaxDZ", 0.2);
    desc.add<double>("GammaCand_EcalclusMinEt", 1.0);
    desc.add<int>("tkminPixelHitsn", 0);
    desc.add<edm::InputTag>("PVProducer", edm::InputTag("offlinePrimaryVertices"));
    desc.add<edm::InputTag>("PFCandidateProducer", edm::InputTag("particleFlow"));
    desc.add<double>("ChargedHadrCand_tkminPt", 0.5);
    desc.add<double>("ChargedHadrCand_tkmaxipt", 0.03);
    desc.add<int>("ChargedHadrCand_tkminPixelHitsn", 0);
    desc.add<bool>("UsePVconstraint", true);
    desc.add<double>("NeutrHadrCand_HcalclusMinEt", 1.0);
    desc.add<edm::InputTag>("PFJetTracksAssociatorProducer", edm::InputTag("ak4PFJetTracksAssociatorAtVertex"));
    desc.add<double>("smearedPVsigmaY", 0.0015);
    desc.add<double>("smearedPVsigmaX", 0.0015);
    desc.add<double>("smearedPVsigmaZ", 0.005);
    desc.add<double>("ChargedHadrCand_tkPVmaxDZ", 0.2);
    desc.add<double>("tkmaxipt", 0.03);
    descriptions.add("pfRecoTauTagInfoProducer", desc);
  }
}

DEFINE_FWK_MODULE(PFRecoTauTagInfoProducer);
