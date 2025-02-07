// -*- C++ -*-
//
// Package:    L1Trigger/L1TTrackMatch
// Class:      L1PhiMesonSelectionProducer
//
/**\class L1PhiMesonSelectionProducer L1PhiMesonSelectionProducer.cc L1Trigger/L1TTrackMatch/plugins/L1PhiMesonSelectionProducer.cc

 Description: Build Phi meson candidates from positively and negatively charged selected L1Tracks (assuming kaons)

 Implementation:
     Inputs:
         std::vector<TTTrack> - Positively and negatively charged collections of selected L1Tracks
     Outputs:
         l1t::TkPhiCandidateCollection - A collection of reconstructed Phi meson candidates

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

// user include files
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefToPtr.h"
#include "DataFormats/L1TCorrelator/interface/TkPhiCandidate.h"
#include "DataFormats/L1TCorrelator/interface/TkPhiCandidateFwd.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1Trigger/interface/Vertex.h"
#include "DataFormats/L1Trigger/interface/VertexWord.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "DataFormats/Math/interface/LorentzVector.h"

//
// class declaration
//

using namespace std;
using namespace edm;
using namespace l1t;

class L1PhiMesonSelectionProducer : public edm::global::EDProducer<> {
public:
  using L1TTTrackType            = TTTrack<Ref_Phase2TrackerDigi_>;
  using TTTrackCollectionType    = std::vector<L1TTTrackType>;
  using TTTrackRef               = edm::Ref<TTTrackCollectionType>;
  using TTTrackRefCollection     = edm::RefVector<TTTrackCollectionType>;
  using TTTrackCollectionHandle  = edm::Handle<TTTrackRefCollection>;
  using TTTrackRefCollectionUPtr = std::unique_ptr<TTTrackRefCollection>;

  explicit L1PhiMesonSelectionProducer(const edm::ParameterSet&);
  ~L1PhiMesonSelectionProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  static constexpr double KaonMass = 0.493677; // GeV
  size_t phiSize = 20;

private:
  // ----------member functions ----------------------
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  // ----------selectors -----------------------------
  // Based on recommendations from https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideGenericSelectors

  // ----------member data ---------------------------
  const edm::EDGetTokenT<TTTrackRefCollection> l1PosKaonTracksToken_;
  const edm::EDGetTokenT<TTTrackRefCollection> l1NegKaonTracksToken_;
  const std::string outputCollectionName_;
  const edm::ParameterSet cutSet_;
  const double tkPairdzMax_, tkPairdRMax_, tkPairMMin_, tkPairMMax_;
  int debug_;
};

//
// constructors and destructor
//
L1PhiMesonSelectionProducer::L1PhiMesonSelectionProducer(const edm::ParameterSet& iConfig)
  : l1PosKaonTracksToken_(consumes<TTTrackRefCollection>(iConfig.getParameter<edm::InputTag>("l1PosKaonTracksInputTag"))),
    l1NegKaonTracksToken_(consumes<TTTrackRefCollection>(iConfig.getParameter<edm::InputTag>("l1NegKaonTracksInputTag"))),
    outputCollectionName_(iConfig.getParameter<std::string>("outputCollectionName")),
    cutSet_(iConfig.getParameter<edm::ParameterSet>("cutSet")),
    tkPairdzMax_(cutSet_.getParameter<double>("tkPairdzMax")),
    tkPairdRMax_(cutSet_.getParameter<double>("tkPairdRMax")),
    tkPairMMin_(cutSet_.getParameter<double>("tkPairMMin")),
    tkPairMMax_(cutSet_.getParameter<double>("tkPairMMax")),
    debug_(iConfig.getParameter<int>("debug")) 
{
  produces<TkPhiCandidateCollection>(outputCollectionName_);
}

L1PhiMesonSelectionProducer::~L1PhiMesonSelectionProducer() {}
// ------------ method called to produce the data  ------------
void L1PhiMesonSelectionProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  auto L1PhiMesonOutput = std::make_unique<l1t::TkPhiCandidateCollection>();

  TTTrackCollectionHandle l1PosKaonTracksHandle;
  iEvent.getByToken(l1PosKaonTracksToken_, l1PosKaonTracksHandle);

  TTTrackCollectionHandle l1NegKaonTracksHandle;
  iEvent.getByToken(l1NegKaonTracksToken_, l1NegKaonTracksHandle);

  size_t nPosKaon = l1PosKaonTracksHandle->size();
  size_t nNegKaon = l1NegKaonTracksHandle->size();
  L1PhiMesonOutput->reserve(phiSize);  

  for (size_t i = 0; i < nPosKaon; ++i) {
    const auto& trackPosKaonRef = l1PosKaonTracksHandle->at(i);
    const auto& trackPosKaon = *trackPosKaonRef;    
    const edm::Ptr<L1TTTrackType>& trackPosKaonReftoPtr = edm::refToPtr(trackPosKaonRef);

    const GlobalVector& trackPosP = trackPosKaon.momentum();
    math::PtEtaPhiMLorentzVector posKaonP4(trackPosP.perp(), trackPosP.eta(), trackPosP.phi(), KaonMass);

    for (size_t j = 0; j < nNegKaon; ++j) {
      const auto& trackNegKaonRef = l1NegKaonTracksHandle->at(j);
      const auto& trackNegKaon = *trackNegKaonRef;

      const edm::Ptr<L1TTTrackType>& trackNegKaonReftoPtr = edm::refToPtr(trackNegKaonRef);      

      const GlobalVector& trackNegP = trackNegKaon.momentum();
      math::PtEtaPhiMLorentzVector negKaonP4(trackNegP.perp(), trackNegP.eta(), trackNegP.phi(), KaonMass);
      
      math::XYZTLorentzVector phiP4(posKaonP4.Px() + negKaonP4.Px(),
				    posKaonP4.Py() + negKaonP4.Py(),
				    posKaonP4.Pz() + negKaonP4.Pz(),
				    posKaonP4.T()  + negKaonP4.T());
    
      TkPhiCandidate tkPhi(phiP4, trackPosKaonReftoPtr, trackNegKaonReftoPtr);

      double dzTrkPair = tkPhi.dzTrkPair();
      if (std::fabs(dzTrkPair) > tkPairdzMax_) continue;
      
      double dRTrkPair = tkPhi.dRTrkPair();
      if (dRTrkPair > tkPairdRMax_) continue;
      
      double mass = tkPhi.p4().M();
      if (mass < tkPairMMin_ || mass > tkPairMMax_) continue;

      bool dupl = false;
      for (const auto& el: *L1PhiMesonOutput) {
	double ptDiff  = el.p4().Pt()  - tkPhi.p4().Pt();
        double etaDiff = el.p4().Eta() - tkPhi.p4().Eta();
        double phiDiff = el.p4().Phi() - tkPhi.p4().Phi();
	if ( fabs(etaDiff) < 1.0e-03 &&
             fabs(phiDiff) < 1.0e-03 &&
             fabs(ptDiff)  < 1.0e-02 )
          {
            dupl = true;
            break;
          }
      }
      if (!dupl) {
	// Put the outputs into the event
	L1PhiMesonOutput->push_back(tkPhi);
      }
    }
  }
  iEvent.put(std::move(L1PhiMesonOutput), outputCollectionName_);
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void L1PhiMesonSelectionProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //L1PhiMesonSelectionProducer
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("l1PosKaonTracksInputTag", edm::InputTag("l1KaonTrackSelectionProducer", "Level1TTKaonTracksSelectedPositivecharge"));
  desc.add<edm::InputTag>("l1NegKaonTracksInputTag", edm::InputTag("l1KaonTrackSelectionProducer", "Level1TTKaonTracksSelectedNegativecharge"));
  desc.add<std::string>("outputCollectionName", "Level1PhiMesonColl");  {
    edm::ParameterSetDescription descCutSet;
    descCutSet.add<double>("tkPairdzMax", 0.5)->setComment("dz between opp. charged track pair must be less than this value, [cm]");
    descCutSet.add<double>("tkPairdRMax", 0.2)->setComment("dR between opp. charged track pair must be less than this value, []");
    descCutSet.add<double>("tkPairMMin", 1.0)->setComment("#track pair mass must be greater than this value, [GeV]");
    descCutSet.add<double>("tkPairMMax", 1.03)->setComment("track pair mass must be less than this value, [GeV]");
    desc.add<edm::ParameterSetDescription>("cutSet", descCutSet);
  }
  desc.add<int>("debug", 0)->setComment("Verbosity levels: 0, 1, 2, 3");
  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1PhiMesonSelectionProducer);
