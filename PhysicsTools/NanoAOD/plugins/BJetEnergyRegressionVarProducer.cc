// -*- C++ -*-
//
// Package:    PhysicsTools/NanoAOD
// Class:      BJetEnergyRegressionVarProducer
//
/**\class BJetEnergyRegressionVarProducer BJetEnergyRegressionVarProducer.cc PhysicsTools/NanoAOD/plugins/BJetEnergyRegressionVarProducer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Marco Peruzzi
//         Created:  Tue, 05 Sep 2017 12:24:38 GMT
//
//

// system include files
#include <memory>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"
#include "RecoVertex/VertexPrimitives/interface/ConvertToFromReco.h"
#include "RecoVertex/VertexPrimitives/interface/VertexState.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"

//
// class declaration
//

template <typename T>
class BJetEnergyRegressionVarProducer : public edm::global::EDProducer<> {
public:
  explicit BJetEnergyRegressionVarProducer(const edm::ParameterSet& iConfig)
      : srcJet_(consumes<edm::View<pat::Jet>>(iConfig.getParameter<edm::InputTag>("src"))),
        srcVtx_(consumes<std::vector<reco::Vertex>>(iConfig.getParameter<edm::InputTag>("pvsrc"))),
        srcSV_(consumes<edm::View<reco::VertexCompositePtrCandidate>>(iConfig.getParameter<edm::InputTag>("svsrc"))) {
    //un prodotto da copiare
    produces<edm::ValueMap<float>>("leptonPtRelv0");
    produces<edm::ValueMap<float>>("leptonPtRelInvv0");  //v0 ~ heppy?
    produces<edm::ValueMap<int>>("leptonPdgId");
    produces<edm::ValueMap<float>>("leptonDeltaR");
    produces<edm::ValueMap<float>>("leadTrackPt");
    produces<edm::ValueMap<float>>("vtxPt");
    produces<edm::ValueMap<float>>("vtxMass");
    produces<edm::ValueMap<float>>("vtx3dL");
    produces<edm::ValueMap<float>>("vtx3deL");
    produces<edm::ValueMap<int>>("vtxNtrk");
    produces<edm::ValueMap<float>>("ptD");
  }
  ~BJetEnergyRegressionVarProducer() override{};

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  std::tuple<float, float, float> calculatePtRatioRelSimple(edm::Ptr<reco::Candidate> lep,
                                                            edm::Ptr<pat::Jet> jet) const;  //old version?

  // ----------member data ---------------------------

  edm::EDGetTokenT<edm::View<pat::Jet>> srcJet_;
  edm::EDGetTokenT<std::vector<reco::Vertex>> srcVtx_;
  edm::EDGetTokenT<edm::View<reco::VertexCompositePtrCandidate>> srcSV_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// member functions
//

// ------------ method called to produce the data  ------------
template <typename T>
void BJetEnergyRegressionVarProducer<T>::produce(edm::StreamID streamID,
                                                 edm::Event& iEvent,
                                                 const edm::EventSetup& iSetup) const {
  const auto& srcJet = iEvent.getHandle(srcJet_);
  const auto& vtxProd = iEvent.get(srcVtx_);
  const auto& svProd = iEvent.get(srcSV_);

  auto nJet = srcJet->size();

  std::vector<float> leptonPtRel_v0(nJet, 0);
  std::vector<float> leptonPtRelInv_v0(nJet, 0);
  std::vector<int> leptonPdgId(nJet, 0);
  std::vector<float> leptonDeltaR(nJet, 0);
  std::vector<float> leadTrackPt(nJet, 0);
  std::vector<float> vtxPt(nJet, 0);
  std::vector<float> vtxMass(nJet, 0);
  std::vector<float> vtx3dL(nJet, 0);
  std::vector<float> vtx3deL(nJet, 0);
  std::vector<int> vtxNtrk(nJet, 0);
  std::vector<float> ptD(nJet, 0);

  const auto& pv = vtxProd.at(0);
  for (unsigned int ij = 0; ij < nJet; ij++) {
    auto jet = srcJet->ptrAt(ij);

    float ptMax = 0;
    float sumWeight = 0;
    float sumPt = 0;

    for (const auto& d : jet->daughterPtrVector()) {
      sumWeight += (d->pt()) * (d->pt());
      sumPt += d->pt();
      if (d->pt() > ptMax)
        ptMax = d->pt();
    }
    leadTrackPt[ij] = ptMax;
    ptD[ij] = (sumWeight > 0 ? sqrt(sumWeight) / sumPt : 0);

    //lepton properties
    float maxLepPt = 0;

    for (const auto& d : jet->daughterPtrVector()) {
      if (abs(d->pdgId()) == 11 || abs(d->pdgId()) == 13) {
        if (d->pt() < maxLepPt)
          continue;
        auto res2 = calculatePtRatioRelSimple(d, jet);
        leptonPtRel_v0[ij] = std::get<1>(res2);
        leptonPtRelInv_v0[ij] = std::get<2>(res2);
        leptonPdgId[ij] = d->pdgId();
        leptonDeltaR[ij] = reco::deltaR(jet->p4(), d->p4());
        maxLepPt = d->pt();
      }
    }

    //Fill vertex properties
    VertexDistance3D vdist;
    float maxFoundSignificance = 0;

    vtxPt[ij] = 0;
    vtxMass[ij] = 0;
    vtx3dL[ij] = 0;
    vtx3deL[ij] = 0;
    vtxNtrk[ij] = 0;

    for (const auto& sv : svProd) {
      GlobalVector flightDir(sv.vertex().x() - pv.x(), sv.vertex().y() - pv.y(), sv.vertex().z() - pv.z());
      GlobalVector jetDir(jet->px(), jet->py(), jet->pz());
      if (reco::deltaR2(flightDir, jetDir) < 0.09) {
        Measurement1D dl = vdist.distance(
            pv, VertexState(RecoVertex::convertPos(sv.position()), RecoVertex::convertError(sv.error())));
        if (dl.significance() > maxFoundSignificance) {
          maxFoundSignificance = dl.significance();
          vtxPt[ij] = sv.pt();
          vtxMass[ij] = sv.p4().M();
          vtx3dL[ij] = dl.value();
          vtx3deL[ij] = dl.error();
          vtxNtrk[ij] = sv.numberOfSourceCandidatePtrs();
        }
      }
    }
  }

  std::unique_ptr<edm::ValueMap<float>> leptonPtRelV_v0(new edm::ValueMap<float>());
  edm::ValueMap<float>::Filler fillerRel_v0(*leptonPtRelV_v0);
  fillerRel_v0.insert(srcJet, leptonPtRel_v0.begin(), leptonPtRel_v0.end());
  fillerRel_v0.fill();
  iEvent.put(std::move(leptonPtRelV_v0), "leptonPtRelv0");

  std::unique_ptr<edm::ValueMap<float>> leptonPtRelInvV_v0(new edm::ValueMap<float>());
  edm::ValueMap<float>::Filler fillerRelInv_v0(*leptonPtRelInvV_v0);
  fillerRelInv_v0.insert(srcJet, leptonPtRelInv_v0.begin(), leptonPtRelInv_v0.end());
  fillerRelInv_v0.fill();
  iEvent.put(std::move(leptonPtRelInvV_v0), "leptonPtRelInvv0");

  std::unique_ptr<edm::ValueMap<float>> leptonDeltaRV(new edm::ValueMap<float>());
  edm::ValueMap<float>::Filler fillerLdR(*leptonDeltaRV);
  fillerLdR.insert(srcJet, leptonDeltaR.begin(), leptonDeltaR.end());
  fillerLdR.fill();
  iEvent.put(std::move(leptonDeltaRV), "leptonDeltaR");

  std::unique_ptr<edm::ValueMap<int>> leptonPtPdgIdV(new edm::ValueMap<int>());
  edm::ValueMap<int>::Filler fillerId(*leptonPtPdgIdV);
  fillerId.insert(srcJet, leptonPdgId.begin(), leptonPdgId.end());
  fillerId.fill();
  iEvent.put(std::move(leptonPtPdgIdV), "leptonPdgId");

  std::unique_ptr<edm::ValueMap<float>> leadTrackPtV(new edm::ValueMap<float>());
  edm::ValueMap<float>::Filler fillerLT(*leadTrackPtV);
  fillerLT.insert(srcJet, leadTrackPt.begin(), leadTrackPt.end());
  fillerLT.fill();
  iEvent.put(std::move(leadTrackPtV), "leadTrackPt");

  std::unique_ptr<edm::ValueMap<float>> vtxPtV(new edm::ValueMap<float>());
  edm::ValueMap<float>::Filler fillerVtxPt(*vtxPtV);
  fillerVtxPt.insert(srcJet, vtxPt.begin(), vtxPt.end());
  fillerVtxPt.fill();
  iEvent.put(std::move(vtxPtV), "vtxPt");

  std::unique_ptr<edm::ValueMap<float>> vtxMassV(new edm::ValueMap<float>());
  edm::ValueMap<float>::Filler fillerVtxMass(*vtxMassV);
  fillerVtxMass.insert(srcJet, vtxMass.begin(), vtxMass.end());
  fillerVtxMass.fill();
  iEvent.put(std::move(vtxMassV), "vtxMass");

  std::unique_ptr<edm::ValueMap<float>> vtx3dLV(new edm::ValueMap<float>());
  edm::ValueMap<float>::Filler fillerVtx3dL(*vtx3dLV);
  fillerVtx3dL.insert(srcJet, vtx3dL.begin(), vtx3dL.end());
  fillerVtx3dL.fill();
  iEvent.put(std::move(vtx3dLV), "vtx3dL");

  std::unique_ptr<edm::ValueMap<float>> vtx3deLV(new edm::ValueMap<float>());
  edm::ValueMap<float>::Filler fillerVtx3deL(*vtx3deLV);
  fillerVtx3deL.insert(srcJet, vtx3deL.begin(), vtx3deL.end());
  fillerVtx3deL.fill();
  iEvent.put(std::move(vtx3deLV), "vtx3deL");

  std::unique_ptr<edm::ValueMap<int>> vtxNtrkV(new edm::ValueMap<int>());
  edm::ValueMap<int>::Filler fillerVtxNT(*vtxNtrkV);
  fillerVtxNT.insert(srcJet, vtxNtrk.begin(), vtxNtrk.end());
  fillerVtxNT.fill();
  iEvent.put(std::move(vtxNtrkV), "vtxNtrk");

  std::unique_ptr<edm::ValueMap<float>> ptDV(new edm::ValueMap<float>());
  edm::ValueMap<float>::Filler fillerPtD(*ptDV);
  fillerPtD.insert(srcJet, ptD.begin(), ptD.end());
  fillerPtD.fill();
  iEvent.put(std::move(ptDV), "ptD");
}

template <typename T>
std::tuple<float, float, float> BJetEnergyRegressionVarProducer<T>::calculatePtRatioRelSimple(
    edm::Ptr<reco::Candidate> lep, edm::Ptr<pat::Jet> jet) const {
  auto lepp4 = lep->p4();
  auto rawp4 = jet->correctedP4("Uncorrected");

  if ((rawp4 - lepp4).R() < 1e-4)
    return std::tuple<float, float, float>(1.0, 0.0, 0.0);

  auto jetp4 = jet->p4();  //(rawp4 - lepp4*(1.0/jet->jecFactor("L1FastJet")))*(jet->pt()/rawp4.pt())+lepp4;
  auto ptratio = lepp4.pt() / jetp4.pt();
  auto ptrel = lepp4.Vect().Cross((jetp4).Vect().Unit()).R();
  auto ptrelinv = jetp4.Vect().Cross((lepp4).Vect().Unit()).R();

  return std::tuple<float, float, float>(ptratio, ptrel, ptrelinv);
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
template <typename T>
void BJetEnergyRegressionVarProducer<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src")->setComment("jet input collection");
  desc.add<edm::InputTag>("pvsrc")->setComment("primary vertex input collection");
  desc.add<edm::InputTag>("svsrc")->setComment("secondary vertex input collection");
  std::string modname;
  if (typeid(T) == typeid(pat::Jet))
    modname += "Jet";
  modname += "RegressionVarProducer";
  descriptions.add(modname, desc);
}

typedef BJetEnergyRegressionVarProducer<pat::Jet> JetRegressionVarProducer;

//define this as a plug-in
DEFINE_FWK_MODULE(JetRegressionVarProducer);
