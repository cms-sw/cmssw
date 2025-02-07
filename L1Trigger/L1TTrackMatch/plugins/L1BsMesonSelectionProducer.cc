// -*- C++ -*-
//
// Package:    L1Trigger/L1TTrackMatch
// Class:      L1BsMesonSelectionProducer
//
/**\class L1BsMesonSelectionProducer L1BsMesonSelectionProducer.cc L1Trigger/L1TTrackMatch/plugins/L1BsMesonSelectionProducer.cc

 Description: Build Bs meson candidates from Phi collection

 Implementation:
     Inputs:
         l1t::TkPhiCandidateCollection - A collection of reconstructed Phi meson candidates
     Outputs:
         l1t::TkBsCandidateCollection - A collection of reconstructed Bs meson candidates

*/
// ----------------------------------------------------------------------------                                                                                                       
// Authors: Alexx Perloff, Pritam Palit (original version, 2021),
//          Sweta Baradia, Suchandra Dutta, Subir Sarkar (January 2025)
//-----------------------------------------------------------------------------  

// system include files
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <TMath.h>
#include <cmath>
#include <array>

// user include files
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefToPtr.h"
#include "DataFormats/L1TCorrelator/interface/TkPhiCandidate.h"
#include "DataFormats/L1TCorrelator/interface/TkBsCandidate.h"
#include "DataFormats/L1TCorrelator/interface/TkPhiCandidateFwd.h"
#include "DataFormats/L1TCorrelator/interface/TkBsCandidateFwd.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include "DataFormats/L1TrackTrigger/interface/TTStub.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"

//
// class declaration

//
using namespace std;
using namespace edm;
using namespace l1t;

class L1BsMesonSelectionProducer : public edm::one::EDProducer<edm::one::SharedResources> {
  //class L1BsMesonSelectionProducer : public edm::global::EDProducer<> {
public:
  using L1TTTrackType            = TTTrack<Ref_Phase2TrackerDigi_>;
  using TTTrackCollection        = std::vector<L1TTTrackType>;
  using TTTrackRefCollection     = edm::RefVector<TTTrackCollection>;
  using TTTrackCollectionHandle  = edm::Handle<TTTrackRefCollection>;
  using L1TTStubCollection =
    std::vector<edm::Ref<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>, TTStub<Ref_Phase2TrackerDigi_>>>;

  explicit L1BsMesonSelectionProducer(const edm::ParameterSet&);
  ~L1BsMesonSelectionProducer() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  static constexpr double KaonMass = 0.493677; // GeV
  size_t bsSize = 10;
    
private:
  // ----------constants, enums and typedefs ---------
  using TkPhiCandidateCollectionHandle = edm::Handle<TkPhiCandidateCollection>;
  // ----------member functions ----------------------
  //void produce(edm::StreamID, edm::Event&, const edm::EventSetup&)  const override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  // ----------selectors -----------------------------
  // Based on recommendations from https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideGenericSelectors

  // ----------member data ---------------------------
  const edm::EDGetTokenT<TkPhiCandidateCollection> l1PhiCandToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;
  const edm::EDGetTokenT<TTTrackRefCollection> posTrackToken_;
  const edm::EDGetTokenT<TTTrackRefCollection> negTrackToken_;

  const std::string outputCollectionName_;
  int debug_;
  const edm::ParameterSet cutSet_;
  const double phiPairdzMax_, phiPairdRMin_, phiPairdRMax_, phiPairMMin_, phiPairMMax_, bsPtMin_;
};

//
// constructors and destructor
//
L1BsMesonSelectionProducer::L1BsMesonSelectionProducer(const edm::ParameterSet& iConfig)
    : l1PhiCandToken_(consumes<TkPhiCandidateCollection>(iConfig.getParameter<edm::InputTag>("l1PhiCandInputTag"))),
    tTopoToken_(esConsumes<TrackerTopology, TrackerTopologyRcd>(edm::ESInputTag("", ""))),
    posTrackToken_(consumes<TTTrackRefCollection>(iConfig.getParameter<edm::InputTag>("posTrackInputTag"))),
    negTrackToken_(consumes<TTTrackRefCollection>(iConfig.getParameter<edm::InputTag>("negTrackInputTag"))),
    outputCollectionName_(iConfig.getParameter<string>("outputCollectionName")),
    debug_(iConfig.getParameter<int>("debug")),
    cutSet_(iConfig.getParameter<edm::ParameterSet>("cutSet")),
    phiPairdzMax_(cutSet_.getParameter<double>("phiPairdzMax")),
    phiPairdRMin_(cutSet_.getParameter<double>("phiPairdRMin")),
    phiPairdRMax_(cutSet_.getParameter<double>("phiPairdRMax")),
    phiPairMMin_(cutSet_.getParameter<double>("phiPairMMin")),
    phiPairMMax_(cutSet_.getParameter<double>("phiPairMMax")),
    bsPtMin_(cutSet_.getParameter<double>("bsPtMin"))
{
  produces<TkBsCandidateCollection>(outputCollectionName_);
}

L1BsMesonSelectionProducer::~L1BsMesonSelectionProducer() {}
// ------------ method called to produce the data  ------------
//void L1BsMesonSelectionProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  void L1BsMesonSelectionProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)  {

  auto L1BsMesonOutput = std::make_unique<TkBsCandidateCollection>();

  TkPhiCandidateCollectionHandle l1PhiCandHandle;
  iEvent.getByToken(l1PhiCandToken_, l1PhiCandHandle);
  size_t nPhiMeson = l1PhiCandHandle->size();
  if (nPhiMeson < 2) return;

  TTTrackCollectionHandle posTrackHandle;
  iEvent.getByToken(posTrackToken_, posTrackHandle);

  TTTrackCollectionHandle negTrackHandle;
  iEvent.getByToken(negTrackToken_, negTrackHandle);


  // Tracker Topology
  const TrackerTopology& tTopo = iSetup.getData(tTopoToken_);
  
  L1BsMesonOutput->reserve(bsSize);

  for (size_t i = 0; i < nPhiMeson; i++) {
    const auto& phiCand1 = l1PhiCandHandle->at(i);
    const math::XYZTLorentzVector& phiCand1P4 = phiCand1.p4();
    double ptPhi1   = phiCand1P4.Pt();
    double etaPhi1  = phiCand1P4.Eta();
    double phiPhi1  = phiCand1P4.Phi();
    double massPhi1 = phiCand1P4.M();       
    math::PtEtaPhiMLorentzVector phi1P4(ptPhi1, etaPhi1, phiPhi1, massPhi1);

    const auto& trka_p = phiCand1.trkPtr(0);
    const auto& trka_n = phiCand1.trkPtr(1);
    double drTrkPhi1 = phiCand1.dRTrkPair();

    for (size_t j = i+1; j < nPhiMeson; j++) {
      const auto& phiCand2 = l1PhiCandHandle->at(j);      
      // Must ensure that the 2 Phi mesons are made of 4 distinct tracks
      // this is non-trivial if the reconstructed Phi mesons do not store reference to the parent tracks
      const auto& trkb_p = phiCand2.trkPtr(0);
      const auto& trkb_n = phiCand2.trkPtr(1);
      if (trka_p == trkb_p || trka_n == trkb_n) continue;

      const math::XYZTLorentzVector& phiCand2P4 = phiCand2.p4();
      double ptPhi2   = phiCand2P4.Pt();
      double etaPhi2  = phiCand2P4.Eta();
      double phiPhi2  = phiCand2P4.Phi();
      double massPhi2 = phiCand2P4.M();       
      math::PtEtaPhiMLorentzVector phi2P4(ptPhi2, etaPhi2, phiPhi2, massPhi2);    
      
      math::PtEtaPhiMLorentzVector tempP4 = phi1P4 + phi2P4;
      math::XYZTLorentzVector bsP4(tempP4.Px(), tempP4.Py(), tempP4.Pz(), tempP4.E());
      TkBsCandidate tkBs(bsP4, phiCand1, phiCand2);      
      double dzPhiPair = tkBs.dzPhiPair();
  
      if (std::fabs(dzPhiPair) > phiPairdzMax_) continue;
        
      double drPhiPair = tkBs.dRPhiPair();
      if (drPhiPair < phiPairdRMin_ || drPhiPair > phiPairdRMax_) continue;
      
      double mass = tkBs.p4().M();
      if (mass < phiPairMMin_ || mass > phiPairMMax_) continue;
      
      if (tkBs.p4().Pt() < bsPtMin_) continue;
      
      L1BsMesonOutput->push_back(tkBs);
    }
  }
  
  // Put the outputs into the event
  iEvent.put(std::move(L1BsMesonOutput), outputCollectionName_);

}
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void L1BsMesonSelectionProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("l1PhiCandInputTag", edm::InputTag("l1PhiMesonSelectionProducer","Level1PhiMesonColl"));
  desc.add<edm::InputTag>("posTrackInputTag", edm::InputTag("l1KaonTrackSelectionProducer","Level1TTKaonTracksSelectedPositivecharge"));
  desc.add<edm::InputTag>("negTrackInputTag", edm::InputTag("l1KaonTrackSelectionProducer","Level1TTKaonTracksSelectedNegativecharge"));
  desc.add<string>("outputCollectionName", "Level1BsHadronicColl");
  desc.add<int>("debug", 0)->setComment("Verbosity levels: 0, 1, 2, 3");
  {
    edm::ParameterSetDescription descCutSet;
    descCutSet.add<double>("phiPairdzMax", 0.5)->setComment("dz between phi pair must be less than this value, [cm]");
    descCutSet.add<double>("phiPairdRMin", 0.2)->setComment("dR between phi pair must be greater than this value");
    descCutSet.add<double>("phiPairdRMax", 1.0)->setComment("dR between phi pair must be less than this value");
    descCutSet.add<double>("phiPairMMin", 5.0)->setComment("Bs mass must be greater than this value, [GeV]");
    descCutSet.add<double>("phiPairMMax", 5.8)->setComment("Bs mass must be less than this value, [GeV]");
    descCutSet.add<double>("bsPtMin", 13.0)->setComment("Bs pT must be greater than this value, [GeV]");
    desc.add<edm::ParameterSetDescription>("cutSet", descCutSet);
  }
  descriptions.addWithDefaultLabel(desc);
}
//define this as a plug-in
DEFINE_FWK_MODULE(L1BsMesonSelectionProducer);
