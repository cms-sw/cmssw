// -*- C++ -*-
//
// Package:    PhysicsTools/NanoAOD
// Class:      LeptonJetVarProducer
// 
/**\class LeptonJetVarProducer LeptonJetVarProducer.cc PhysicsTools/NanoAOD/plugins/LeptonJetVarProducer.cc

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

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"

#include "PhysicsTools/NanoAOD/interface/MatchingUtils.h"

//
// class declaration
//

template <typename T>
class LeptonJetVarProducer : public edm::global::EDProducer<> {
   public:
  explicit LeptonJetVarProducer(const edm::ParameterSet &iConfig):
    srcJet_(consumes<edm::View<pat::Jet>>(iConfig.getParameter<edm::InputTag>("srcJet"))),
    srcLep_(consumes<edm::View<T>>(iConfig.getParameter<edm::InputTag>("srcLep"))),
    srcVtx_(consumes<std::vector<reco::Vertex>>(iConfig.getParameter<edm::InputTag>("srcVtx")))
  {
    produces<edm::ValueMap<float>>("ptRatio");
    produces<edm::ValueMap<float>>("ptRel");
    produces<edm::ValueMap<float>>("jetNDauChargedMVASel");
    produces<edm::ValueMap<reco::CandidatePtr>>("jetForLepJetVar");
  }
  ~LeptonJetVarProducer() override {};

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  std::tuple<float,float,float> calculatePtRatioRel(edm::Ptr<reco::Candidate> lep, edm::Ptr<pat::Jet> jet, const reco::Vertex &vtx) const;

      // ----------member data ---------------------------

  edm::EDGetTokenT<edm::View<pat::Jet>> srcJet_;
  edm::EDGetTokenT<edm::View<T>> srcLep_;
  edm::EDGetTokenT<std::vector<reco::Vertex>> srcVtx_;
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
void
LeptonJetVarProducer<T>::produce(edm::StreamID streamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const
{

  edm::Handle<edm::View<pat::Jet>> srcJet;
  iEvent.getByToken(srcJet_, srcJet);
  edm::Handle<edm::View<T>> srcLep;
  iEvent.getByToken(srcLep_, srcLep);
  edm::Handle<std::vector<reco::Vertex>> srcVtx;
  iEvent.getByToken(srcVtx_, srcVtx);

  unsigned int nJet = srcJet->size();
  unsigned int nLep = srcLep->size();

  std::vector<float> ptRatio(nLep,-1);
  std::vector<float> ptRel(nLep,-1);
  std::vector<float> jetNDauChargedMVASel(nLep,0);
  std::vector<reco::CandidatePtr> jetForLepJetVar(nLep,reco::CandidatePtr());

  const auto & pv = (*srcVtx)[0];

  for (unsigned int il = 0; il<nLep; il++){
    for (unsigned int ij = 0; ij<nJet; ij++){
      auto lep = srcLep->ptrAt(il);
      auto jet = srcJet->ptrAt(ij);
      if(matchByCommonSourceCandidatePtr(*lep,*jet)){
	  auto res = calculatePtRatioRel(lep,jet,pv);
	  ptRatio[il] = std::get<0>(res);
	  ptRel[il] = std::get<1>(res);
	  jetNDauChargedMVASel[il] = std::get<2>(res);
	  jetForLepJetVar[il] = jet;
	  break; // take leading jet with shared source candidates
	}
    }
  }

  std::unique_ptr<edm::ValueMap<float>> ptRatioV(new edm::ValueMap<float>());
  edm::ValueMap<float>::Filler fillerRatio(*ptRatioV);
  fillerRatio.insert(srcLep,ptRatio.begin(),ptRatio.end());
  fillerRatio.fill();
  iEvent.put(std::move(ptRatioV),"ptRatio");

  std::unique_ptr<edm::ValueMap<float>> ptRelV(new edm::ValueMap<float>());
  edm::ValueMap<float>::Filler fillerRel(*ptRelV);
  fillerRel.insert(srcLep,ptRel.begin(),ptRel.end());
  fillerRel.fill();
  iEvent.put(std::move(ptRelV),"ptRel");

  std::unique_ptr<edm::ValueMap<float>> jetNDauChargedMVASelV(new edm::ValueMap<float>());
  edm::ValueMap<float>::Filler fillerNDau(*jetNDauChargedMVASelV);
  fillerNDau.insert(srcLep,jetNDauChargedMVASel.begin(),jetNDauChargedMVASel.end());
  fillerNDau.fill();
  iEvent.put(std::move(jetNDauChargedMVASelV),"jetNDauChargedMVASel");

  std::unique_ptr<edm::ValueMap<reco::CandidatePtr>> jetForLepJetVarV(new edm::ValueMap<reco::CandidatePtr>());
  edm::ValueMap<reco::CandidatePtr>::Filler fillerjetForLepJetVar(*jetForLepJetVarV);
  fillerjetForLepJetVar.insert(srcLep,jetForLepJetVar.begin(),jetForLepJetVar.end());
  fillerjetForLepJetVar.fill();
  iEvent.put(std::move(jetForLepJetVarV),"jetForLepJetVar");


}

template <typename T>
std::tuple<float,float,float>
LeptonJetVarProducer<T>::calculatePtRatioRel(edm::Ptr<reco::Candidate> lep, edm::Ptr<pat::Jet> jet, const reco::Vertex &vtx) const {
 
  auto rawp4 = jet->correctedP4("Uncorrected");
  auto lepp4 = lep->p4();

  if ((rawp4-lepp4).R()<1e-4) return std::tuple<float,float,float>(1.0,0.0,0.0);

  auto l1corrFactor = jet->jecFactor("L1FastJet")/jet->jecFactor("Uncorrected");

  auto jetp4 = (rawp4 - lepp4*(1.0/l1corrFactor))*(jet->pt()/rawp4.pt())+lepp4;
  auto ptratio = lepp4.pt()/jetp4.pt();
  auto ptrel = lepp4.Vect().Cross((jetp4-lepp4).Vect().Unit()).R();

  unsigned int jndau = 0;
  for(const auto _d : jet->daughterPtrVector()) {
    const auto d = dynamic_cast<const pat::PackedCandidate*>(_d.get());
    if (d->charge()==0) continue;
    if (d->fromPV()<=1) continue;
    if (deltaR(*d,*lep)>0.4) continue;
    if (!(d->hasTrackDetails())) continue;
    auto tk = d->pseudoTrack();
    if(tk.pt()>1 &&
       tk.hitPattern().numberOfValidHits()>=8 &&
       tk.hitPattern().numberOfValidPixelHits()>=2 &&
       tk.normalizedChi2()<5 &&
       fabs(tk.dxy(vtx.position()))<0.2 &&
       fabs(tk.dz(vtx.position()))<17
       ) jndau++;
  }

  return std::tuple<float,float,float>(ptratio,ptrel,float(jndau));
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
template <typename T>
void
LeptonJetVarProducer<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("srcJet")->setComment("jet input collection");
  desc.add<edm::InputTag>("srcLep")->setComment("lepton input collection");
  desc.add<edm::InputTag>("srcVtx")->setComment("primary vertex input collection");
  std::string modname;
  if (typeid(T) == typeid(pat::Muon)) modname+="Muon";
  else if (typeid(T) == typeid(pat::Electron)) modname+="Electron";
  modname+="JetVarProducer";
  descriptions.add(modname,desc);
}

typedef LeptonJetVarProducer<pat::Muon> MuonJetVarProducer;
typedef LeptonJetVarProducer<pat::Electron> ElectronJetVarProducer;

//define this as a plug-in
DEFINE_FWK_MODULE(MuonJetVarProducer);
DEFINE_FWK_MODULE(ElectronJetVarProducer);
