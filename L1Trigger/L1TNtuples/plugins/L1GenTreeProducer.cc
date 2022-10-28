// -*- C++ -*-
//
// Package:    L1TriggerDPG/L1Ntuples
// Class:      L1GenTreeProducer
//
/**\class L1GenTreeProducer L1GenTreeProducer.cc L1TriggerDPG/L1Ntuples/src/L1GenTreeProducer.cc

Description: Produce L1 Extra tree

Implementation:
     
*/
//
// Original Author:
//         Created:
// $Id: L1GenTreeProducer.cc,v 1.8 2012/08/29 12:44:03 jbrooke Exp $
//
//

// system include files
#include <memory>

// framework
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// input data formats
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "HepMC/GenParticle.h"
#include "HepMC/GenVertex.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"

// ROOT output stuff
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TTree.h"

#include "L1Trigger/L1TNtuples/interface/L1AnalysisGeneratorDataFormat.h"

//
// class declaration
//

class L1GenTreeProducer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit L1GenTreeProducer(const edm::ParameterSet&);
  ~L1GenTreeProducer() override;

private:
  void beginJob(void) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

private:
  unsigned maxL1Upgrade_;

  // output file
  edm::Service<TFileService> fs_;

  // tree
  TTree* tree_;

  // data format
  std::unique_ptr<L1Analysis::L1AnalysisGeneratorDataFormat> l1GenData_;

  // EDM input tags
  edm::EDGetTokenT<reco::GenJetCollection> genJetToken_;
  edm::EDGetTokenT<reco::GenParticleCollection> genParticleToken_;
  edm::EDGetTokenT<std::vector<PileupSummaryInfo>> pileupInfoToken_;
  edm::EDGetTokenT<GenEventInfoProduct> genInfoToken_;
};

L1GenTreeProducer::L1GenTreeProducer(const edm::ParameterSet& iConfig) {
  genJetToken_ = consumes<reco::GenJetCollection>(iConfig.getUntrackedParameter<edm::InputTag>("genJetToken"));
  genParticleToken_ =
      consumes<reco::GenParticleCollection>(iConfig.getUntrackedParameter<edm::InputTag>("genParticleToken"));
  pileupInfoToken_ =
      consumes<std::vector<PileupSummaryInfo>>(iConfig.getUntrackedParameter<edm::InputTag>("pileupInfoToken"));
  genInfoToken_ = consumes<GenEventInfoProduct>(iConfig.getParameter<edm::InputTag>("genInfoToken"));

  l1GenData_ = std::make_unique<L1Analysis::L1AnalysisGeneratorDataFormat>();

  usesResource(TFileService::kSharedResource);
  // set up output
  tree_ = fs_->make<TTree>("L1GenTree", "L1GenTree");
  tree_->Branch("Generator", "L1Analysis::L1AnalysisGeneratorDataFormat", l1GenData_.get(), 32000, 3);
}

L1GenTreeProducer::~L1GenTreeProducer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to for each event  ------------
void L1GenTreeProducer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<GenEventInfoProduct> genInfo;
  iEvent.getByToken(genInfoToken_, genInfo);

  l1GenData_->Reset();

  if (genInfo.isValid()) {
    l1GenData_->weight = genInfo->weight();
    l1GenData_->pthat = genInfo->hasBinningValues() ? (genInfo->binningValues())[0] : 0.0;
  }

  edm::Handle<reco::GenJetCollection> genJets;
  iEvent.getByToken(genJetToken_, genJets);

  if (genJets.isValid()) {
    reco::GenJetCollection::const_iterator jetItr = genJets->begin();
    reco::GenJetCollection::const_iterator jetEnd = genJets->end();
    for (; jetItr != jetEnd; ++jetItr) {
      l1GenData_->jetPt.push_back(jetItr->pt());
      l1GenData_->jetEta.push_back(jetItr->eta());
      l1GenData_->jetPhi.push_back(jetItr->phi());
      l1GenData_->nJet++;
    }

  } else {
    edm::LogWarning("MissingProduct") << "Gen jets not found. Branch will not be filled" << std::endl;
  }

  edm::Handle<reco::GenParticleCollection> genParticles;
  iEvent.getByToken(genParticleToken_, genParticles);

  if (genParticles.isValid()) {
    int nPart{0};

    for (size_t i = 0; i < genParticles->size(); ++i) {
      const reco::GenParticle& p = (*genParticles)[i];
      int id = p.pdgId();

      // See if the parent was interesting
      int parentID = -10000;
      unsigned int nMo = p.numberOfMothers();
      for (unsigned int i = 0; i < nMo; ++i) {
        int thisParentID = dynamic_cast<const reco::GenParticle*>(p.mother(i))->pdgId();
        //
        // Is this a bottom hadron?
        int hundredsIndex = abs(thisParentID) / 100;
        int thousandsIndex = abs(thisParentID) / 1000;
        if (((abs(thisParentID) >= 23) && (abs(thisParentID) <= 25)) || (abs(thisParentID) == 6) ||
            (hundredsIndex == 5) || (hundredsIndex == 4) || (thousandsIndex == 5) || (thousandsIndex == 4))
          parentID = thisParentID;
      }
      if ((parentID == -10000) && (nMo > 0))
        parentID = dynamic_cast<const reco::GenParticle*>(p.mother(0))->pdgId();
      //
      // If the parent of this particle is interesting, store all of the info
      if ((parentID != p.pdgId()) &&
          ((parentID > -9999) || (abs(id) == 11) || (abs(id) == 13) || (abs(id) == 23) || (abs(id) == 24) ||
           (abs(id) == 25) || (abs(id) == 4) || (abs(id) == 5) || (abs(id) == 6))) {
        l1GenData_->partId.push_back(p.pdgId());
        l1GenData_->partStat.push_back(p.status());
        l1GenData_->partPt.push_back(p.pt());
        l1GenData_->partEta.push_back(p.eta());
        l1GenData_->partPhi.push_back(p.phi());
        l1GenData_->partE.push_back(p.energy());
        l1GenData_->partParent.push_back(parentID);
        l1GenData_->partCh.push_back(p.charge());
        ++nPart;
      }
    }
    l1GenData_->nPart = nPart;
  }

  edm::Handle<std::vector<PileupSummaryInfo>> puInfoCollection;
  iEvent.getByToken(pileupInfoToken_, puInfoCollection);

  if (!puInfoCollection.isValid()) {
    throw cms::Exception("ProductNotValid") << "pileupInfoSource not valid";
  }

  // Loop over vector, find in-time entry, then store the relevant info
  std::vector<PileupSummaryInfo>::const_iterator puItr = puInfoCollection->begin();
  std::vector<PileupSummaryInfo>::const_iterator puEnd = puInfoCollection->end();
  for (; puItr != puEnd; ++puItr) {
    int bx = puItr->getBunchCrossing();
    if (bx == 0) {
      l1GenData_->nMeanPU = puItr->getTrueNumInteractions();
      l1GenData_->nVtx = puItr->getPU_NumInteractions();
      break;
    }
  }

  tree_->Fill();
}

// ------------ method called once each job just before starting event loop  ------------
void L1GenTreeProducer::beginJob(void) {}

// ------------ method called once each job just after ending the event loop  ------------
void L1GenTreeProducer::endJob() {}

//define this as a plug-in
DEFINE_FWK_MODULE(L1GenTreeProducer);
