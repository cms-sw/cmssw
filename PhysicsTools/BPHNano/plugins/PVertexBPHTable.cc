// -*- C++ -*-
//
// Package:    PhysicsTools/NanoAOD
// Class:      PVertexBPHTable
//
/*
 Simple primary vertex table  
 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  gkaratha
//         Created:  Mon, 28 Aug 2024 09:26:39 GMT
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"
#include "RecoVertex/VertexTools/interface/VertexDistanceXY.h"
#include "RecoVertex/VertexPrimitives/interface/ConvertToFromReco.h"
#include "RecoVertex/VertexPrimitives/interface/VertexState.h"
#include "DataFormats/Common/interface/ValueMap.h"

//
// class declaration
//

class PVertexBPHTable : public edm::stream::EDProducer<> {
public:
  explicit PVertexBPHTable(const edm::ParameterSet&);
  ~PVertexBPHTable() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginStream(edm::StreamID) override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endStream() override;


  // ----------member data ---------------------------

  const edm::EDGetTokenT<std::vector<reco::Vertex>> pvs_;
  const edm::EDGetTokenT<edm::ValueMap<float>> pvsScore_;
  const StringCutObjectSelector<reco::Vertex> goodPvCut_;
  const std::string pvName_;
};

//
// constructors and destructor
//
PVertexBPHTable::PVertexBPHTable(const edm::ParameterSet& params)
    : pvs_(consumes<std::vector<reco::Vertex>>(params.getParameter<edm::InputTag>("pvSrc"))),
      pvsScore_(consumes<edm::ValueMap<float>>(params.getParameter<edm::InputTag>("pvSrc"))),
      goodPvCut_(params.getParameter<std::string>("goodPvCut"), true),
      pvName_(params.getParameter<std::string>("pvName"))

{
  produces<nanoaod::FlatTable>("pv");
  produces<edm::PtrVector<reco::Candidate>>();
}

PVertexBPHTable::~PVertexBPHTable() {
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------

void PVertexBPHTable::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  const auto& pvsScoreProd = iEvent.get(pvsScore_);
  auto pvsCol = iEvent.getHandle(pvs_);

  auto selCandPv = std::make_unique<PtrVector<reco::Candidate>>();
  std::vector<float> pvscore, chi2, covXX, covYY, covZZ, covXY, covXZ, covYZ,
                     vx, vy, vz, pt, eta, phi, mass, ndof;
  std::vector<int> charge, ntracks;

  size_t i=0;
  for (const auto& pv : *pvsCol){
    if (!goodPvCut_(pv)){
       i++;
       continue;
    }
    int sum_charge = 0;
    pvscore.push_back(pvsScoreProd.get(pvsCol.id(), i));
    ntracks.push_back(pv.tracksSize() );
    chi2.push_back(pv.chi2() ); 
    covXX.push_back(pv.covariance(0,0)); 
    covYY.push_back(pv.covariance(1,1));
    covZZ.push_back(pv.covariance(2,2)); 
    covXY.push_back(pv.covariance(0,1)); 
    covXZ.push_back(pv.covariance(0,2)); 
    covYZ.push_back(pv.covariance(1,2));
    vx.push_back(pv.x());
    vy.push_back(pv.y()); 
    vz.push_back(pv.z()); 
    pt.push_back(pv.p4().pt());
    eta.push_back(pv.p4().eta());  
    phi.push_back(pv.p4().phi()); 
    mass.push_back(pv.p4().M());
    ndof.push_back(pv.ndof());
    i++;
   
  }
  auto table = std::make_unique<nanoaod::FlatTable>(pvscore.size(), pvName_, false,false);
  table->addColumn<float>("score", pvscore, "", 10);
  table->addColumn<float>("vx", vx, "", 10);
  table->addColumn<float>("vy", vy, "", 10);
  table->addColumn<float>("vz", vz, "", 10);
  table->addColumn<float>("pt", pt, "", 10);
  table->addColumn<float>("eta", eta, "", 10);
  table->addColumn<float>("phi", phi, "", 10);
  table->addColumn<float>("mass", mass, "", 10);
  table->addColumn<float>("chi2", chi2, "", 10);
  table->addColumn<float>("ndof", ndof, "", 10);
  table->addColumn<float>("covXX", covXX, "", 10);
  table->addColumn<float>("covYY", covYY, "", 10);
  table->addColumn<float>("covZZ", covZZ, "", 10);
  table->addColumn<float>("covXY", covXY, "", 10);
  table->addColumn<float>("covXZ", covXZ, "", 10);
  table->addColumn<float>("covYZ", covYZ, "", 10);
  table->addColumn<uint8_t>("ntracks", ntracks, "");


  iEvent.put(std::move(table), "pv");
}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void PVertexBPHTable::beginStream(edm::StreamID) {}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void PVertexBPHTable::endStream() {}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void PVertexBPHTable::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("pvSrc")->setComment(
      "std::vector<reco::Vertex> and ValueMap<float> primary vertex input collections");
  desc.add<std::string>("goodPvCut")->setComment("selection on the primary vertex");

  desc.add<std::string>("pvName")->setComment("name of the flat table ouput");

  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(PVertexBPHTable);
