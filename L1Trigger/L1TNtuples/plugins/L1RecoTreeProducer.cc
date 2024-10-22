// -*- C++ -*-
//
// Package:    L1Trigger/L1TNtuples
// Class:      L1RecoTreeProducer
//
/**\class L1RecoTreeProducer L1RecoTreeProducer.cc L1Trigger/L1TNtuples/src/L1RecoTreeProducer.cc
 Description: Produces tree containing reco quantities
*/

// system include files
#include <memory>

// framework
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"

// ROOT output stuff
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TH1.h"
#include "TTree.h"
#include "TF1.h"

// EDM formats
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

//local  data formats
#include "L1Trigger/L1TNtuples/interface/L1AnalysisRecoVertexDataFormat.h"

//
// class declaration
//

class L1RecoTreeProducer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit L1RecoTreeProducer(const edm::ParameterSet&);
  ~L1RecoTreeProducer() override;

private:
  void beginJob(void) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

public:
  L1Analysis::L1AnalysisRecoVertexDataFormat* vtxData_;

private:
  // output file
  edm::Service<TFileService> fs_;

  // tree
  TTree* tree_;

  // EDM input tags
  edm::EDGetTokenT<reco::VertexCollection> vtxToken_;

  unsigned int maxVtx_;
};

L1RecoTreeProducer::L1RecoTreeProducer(const edm::ParameterSet& iConfig) {
  vtxToken_ = consumes<reco::VertexCollection>(
      iConfig.getUntrackedParameter("vtxToken", edm::InputTag("offlinePrimaryVertices")));

  maxVtx_ = iConfig.getParameter<unsigned int>("maxVtx");

  vtxData_ = new L1Analysis::L1AnalysisRecoVertexDataFormat();

  usesResource(TFileService::kSharedResource);
  // set up output
  tree_ = fs_->make<TTree>("RecoTree", "RecoTree");
  tree_->Branch("Vertex", "L1Analysis::L1AnalysisRecoVertexDataFormat", &vtxData_, 32000, 3);
}

L1RecoTreeProducer::~L1RecoTreeProducer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to for each event  ------------
void L1RecoTreeProducer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  vtxData_->Reset();

  // get vertices
  edm::Handle<reco::VertexCollection> vertices;
  iEvent.getByToken(vtxToken_, vertices);

  if (vertices.isValid()) {
    for (reco::VertexCollection::const_iterator it = vertices->begin();
         it != vertices->end() && vtxData_->nVtx < maxVtx_;
         ++it) {
      if (!it->isFake()) {
        vtxData_->NDoF.push_back(it->ndof());
        vtxData_->Z.push_back(it->z());
        vtxData_->Rho.push_back(it->position().rho());
        vtxData_->nVtx++;
      }
    }
    tree_->Fill();
  }
}

// ------------ method called once each job just before starting event loop  ------------
void L1RecoTreeProducer::beginJob(void) {}

// ------------ method called once each job just after ending the event loop  ------------
void L1RecoTreeProducer::endJob() {}

//define this as a plug-in
DEFINE_FWK_MODULE(L1RecoTreeProducer);
