// -*- C++ -*-
//
// Package:    PhysicsTools/PatAlgos
// Class:      PATHemisphereProducer
//
/**\class PATHemisphereProducer PATHemisphereProducer.cc "PhysicsTools/PatAlgos/plugins/PATHemisphereProducer.cc"

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Authors:  Christian Autermann, Tanja Rommerskirchen
//          Created:  Sat Mar 22 12:58:04 CET 2008
//
//

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/Hemisphere.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Candidate/interface/Particle.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "PhysicsTools/UtilAlgos/interface/ParameterAdapter.h"
#include "PhysicsTools/PatAlgos/interface/HemisphereAlgo.h"

#include <map>
#include <memory>
#include <utility>
#include <vector>

class PATHemisphereProducer : public edm::global::EDProducer<> {
public:
  explicit PATHemisphereProducer(const edm::ParameterSet&);
  ~PATHemisphereProducer() override;

  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  // ----------member data ---------------------------
  /// Input: All PAT objects that are to cross-clean  or needed for that
  const edm::EDGetTokenT<reco::CandidateView> _patJetsToken;
  //       edm::EDGetTokenT<reco::CandidateView> _patMetsToken;
  const edm::EDGetTokenT<reco::CandidateView> _patMuonsToken;
  const edm::EDGetTokenT<reco::CandidateView> _patElectronsToken;
  const edm::EDGetTokenT<reco::CandidateView> _patPhotonsToken;
  const edm::EDGetTokenT<reco::CandidateView> _patTausToken;

  const float _minJetEt;
  const float _minMuonEt;
  const float _minElectronEt;
  const float _minTauEt;
  const float _minPhotonEt;

  const float _maxJetEta;
  const float _maxMuonEta;
  const float _maxElectronEta;
  const float _maxTauEta;
  const float _maxPhotonEta;

  const int _seedMethod;
  const int _combinationMethod;

  typedef std::vector<float> HemiAxis;
};

using namespace pat;

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
PATHemisphereProducer::PATHemisphereProducer(const edm::ParameterSet& iConfig)
    : _patJetsToken(consumes<reco::CandidateView>(iConfig.getParameter<edm::InputTag>("patJets"))),
      _patMuonsToken(consumes<reco::CandidateView>(iConfig.getParameter<edm::InputTag>("patMuons"))),
      _patElectronsToken(consumes<reco::CandidateView>(iConfig.getParameter<edm::InputTag>("patElectrons"))),
      _patPhotonsToken(consumes<reco::CandidateView>(iConfig.getParameter<edm::InputTag>("patPhotons"))),
      _patTausToken(consumes<reco::CandidateView>(iConfig.getParameter<edm::InputTag>("patTaus"))),

      _minJetEt(iConfig.getParameter<double>("minJetEt")),
      _minMuonEt(iConfig.getParameter<double>("minMuonEt")),
      _minElectronEt(iConfig.getParameter<double>("minElectronEt")),
      _minTauEt(iConfig.getParameter<double>("minTauEt")),
      _minPhotonEt(iConfig.getParameter<double>("minPhotonEt")),

      _maxJetEta(iConfig.getParameter<double>("maxJetEta")),
      _maxMuonEta(iConfig.getParameter<double>("maxMuonEta")),
      _maxElectronEta(iConfig.getParameter<double>("maxElectronEta")),
      _maxTauEta(iConfig.getParameter<double>("maxTauEta")),
      _maxPhotonEta(iConfig.getParameter<double>("maxPhotonEta")),

      _seedMethod(iConfig.getParameter<int>("seedMethod")),
      _combinationMethod(iConfig.getParameter<int>("combinationMethod"))

{
  produces<std::vector<pat::Hemisphere>>();
}

PATHemisphereProducer::~PATHemisphereProducer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void PATHemisphereProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  using namespace edm;
  using namespace std;

  std::vector<float> vPx, vPy, vPz, vE;
  std::vector<float> vA1, vA2;
  std::vector<int> vgroups;
  std::vector<reco::CandidatePtr> componentPtrs;

  //Jets
  Handle<reco::CandidateView> pJets;
  iEvent.getByToken(_patJetsToken, pJets);

  //Muons
  Handle<reco::CandidateView> pMuons;
  iEvent.getByToken(_patMuonsToken, pMuons);

  //Electrons
  Handle<reco::CandidateView> pElectrons;
  iEvent.getByToken(_patElectronsToken, pElectrons);

  //Photons
  Handle<reco::CandidateView> pPhotons;
  iEvent.getByToken(_patPhotonsToken, pPhotons);

  //Taus
  Handle<reco::CandidateView> pTaus;
  iEvent.getByToken(_patTausToken, pTaus);

  //fill e,p vector with information from all objects (hopefully cleaned before)
  for (int i = 0; i < (int)(*pJets).size(); i++) {
    if ((*pJets)[i].pt() < _minJetEt || fabs((*pJets)[i].eta()) > _maxJetEta)
      continue;

    componentPtrs.push_back(pJets->ptrAt(i));
  }

  for (int i = 0; i < (int)(*pMuons).size(); i++) {
    if ((*pMuons)[i].pt() < _minMuonEt || fabs((*pMuons)[i].eta()) > _maxMuonEta)
      continue;

    componentPtrs.push_back(pMuons->ptrAt(i));
  }

  for (int i = 0; i < (int)(*pElectrons).size(); i++) {
    if ((*pElectrons)[i].pt() < _minElectronEt || fabs((*pElectrons)[i].eta()) > _maxElectronEta)
      continue;

    componentPtrs.push_back(pElectrons->ptrAt(i));
  }

  for (int i = 0; i < (int)(*pPhotons).size(); i++) {
    if ((*pPhotons)[i].pt() < _minPhotonEt || fabs((*pPhotons)[i].eta()) > _maxPhotonEta)
      continue;

    componentPtrs.push_back(pPhotons->ptrAt(i));
  }

  //aren't taus included in jets?
  for (int i = 0; i < (int)(*pTaus).size(); i++) {
    if ((*pTaus)[i].pt() < _minTauEt || fabs((*pTaus)[i].eta()) > _maxTauEta)
      continue;

    componentPtrs.push_back(pTaus->ptrAt(i));
  }

  // create product
  auto hemispheres = std::make_unique<std::vector<Hemisphere>>();
  hemispheres->reserve(2);

  //calls HemiAlgorithm for seed method 3 (transv. inv. Mass) and association method 3 (Lund algo)
  HemisphereAlgo myHemi(componentPtrs, _seedMethod, _combinationMethod);

  //get Hemisphere Axis
  vA1 = myHemi.getAxis1();
  vA2 = myHemi.getAxis2();

  reco::Particle::LorentzVector p1(vA1[0] * vA1[3], vA1[1] * vA1[3], vA1[2] * vA1[3], vA1[4]);
  hemispheres->push_back(Hemisphere(p1));

  reco::Particle::LorentzVector p2(vA2[0] * vA2[3], vA2[1] * vA2[3], vA2[2] * vA2[3], vA2[4]);
  hemispheres->push_back(Hemisphere(p2));

  //get information to which Hemisphere each object belongs
  vgroups = myHemi.getGrouping();

  for (unsigned int i = 0; i < vgroups.size(); ++i) {
    if (vgroups[i] == 1) {
      (*hemispheres)[0].addDaughter(componentPtrs[i]);
    } else {
      (*hemispheres)[1].addDaughter(componentPtrs[i]);
    }
  }

  iEvent.put(std::move(hemispheres));
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PATHemisphereProducer);
