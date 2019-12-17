// -*- C++ -*-
//
// Package: RecoPixelVertexing/PixelVertexFinding
// Class: PixelVertexCollectionTrimmer
//
/**\class PixelVertexCollectionTrimmer PixelVertexCollectionTrimmer.cc RecoPixelVertexing/PixelVertexFinding/src/PixelVertexCollectionTrimmer.cc

Description: [one line class summary]

Implementation:
[Notes on implementation]
*/
//
// Original Author: Riccardo Manzoni
// Created: Tue, 01 Apr 2014 10:11:16 GMT
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

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "RecoPixelVertexing/PixelVertexFinding/interface/PVClusterComparer.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

class PixelVertexCollectionTrimmer : public edm::stream::EDProducer<> {
public:
  explicit PixelVertexCollectionTrimmer(const edm::ParameterSet&);
  ~PixelVertexCollectionTrimmer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  edm::EDGetTokenT<reco::VertexCollection> vtxToken_;
  unsigned int maxVtx_;
  double fractionSumPt2_;
  double minSumPt2_;

  PVClusterComparer* pvComparer_;
};

PixelVertexCollectionTrimmer::PixelVertexCollectionTrimmer(const edm::ParameterSet& iConfig) {
  edm::InputTag vtxInputTag = iConfig.getParameter<edm::InputTag>("src");
  vtxToken_ = consumes<reco::VertexCollection>(vtxInputTag);
  maxVtx_ = iConfig.getParameter<unsigned int>("maxVtx");
  fractionSumPt2_ = iConfig.getParameter<double>("fractionSumPt2");
  minSumPt2_ = iConfig.getParameter<double>("minSumPt2");

  edm::ParameterSet PVcomparerPSet = iConfig.getParameter<edm::ParameterSet>("PVcomparer");
  double track_pt_min = PVcomparerPSet.getParameter<double>("track_pt_min");
  double track_pt_max = PVcomparerPSet.getParameter<double>("track_pt_max");
  double track_chi2_max = PVcomparerPSet.getParameter<double>("track_chi2_max");
  double track_prob_min = PVcomparerPSet.getParameter<double>("track_prob_min");

  pvComparer_ = new PVClusterComparer(track_pt_min, track_pt_max, track_chi2_max, track_prob_min);

  produces<reco::VertexCollection>();
}

PixelVertexCollectionTrimmer::~PixelVertexCollectionTrimmer() {}

// ------------ method called to produce the data ------------
void PixelVertexCollectionTrimmer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  auto vtxs_trim = std::make_unique<reco::VertexCollection>();

  edm::Handle<reco::VertexCollection> vtxs;
  iEvent.getByToken(vtxToken_, vtxs);

  double sumpt2;
  //double sumpt2previous = -99. ;

  // this is not the logic we want, at least for now
  // if requires the sumpt2 for vtx_n to be > threshold * sumpt2 vtx_n-1
  // for (reco::VertexCollection::const_iterator vtx = vtxs->begin(); vtx != vtxs->end(); ++vtx, ++counter){
  // if (counter > maxVtx_) break ;
  // sumpt2 = PVCluster.pTSquaredSum(*vtx) ;
  // if (sumpt2 > sumpt2previous*fractionSumPt2_ && sumpt2 > minSumPt2_ ) vtxs_trim->push_back(*vtx) ;
  // else if (counter == 0 ) vtxs_trim->push_back(*vtx) ;
  // sumpt2previous = sumpt2 ;
  // }

  double sumpt2first = pvComparer_->pTSquaredSum(*(vtxs->begin()));

  for (reco::VertexCollection::const_iterator vtx = vtxs->begin(), evtx = vtxs->end(); vtx != evtx; ++vtx) {
    if (vtxs_trim->size() >= maxVtx_)
      break;
    sumpt2 = pvComparer_->pTSquaredSum(*vtx);
    //    std::cout << "sumpt2: " << sumpt2 << "[" << sumpt2first << "]" << std::endl;
    //    if (sumpt2 >= sumpt2first*fractionSumPt2_ && sumpt2 > minSumPt2_ ) vtxs_trim->push_back(*vtx) ;
    if (sumpt2 >= sumpt2first * fractionSumPt2_ && sumpt2 > minSumPt2_)
      vtxs_trim->push_back(*vtx);
  }
  //  std::cout << " ==> # vertices: " << vtxs_trim->size() << std::endl;
  iEvent.put(std::move(vtxs_trim));
}

// ------------ method fills 'descriptions' with the allowed parameters for the module ------------
void PixelVertexCollectionTrimmer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag(""))->setComment("input (pixel) vertex collection");
  desc.add<unsigned int>("maxVtx", 100)->setComment("max output collection size (number of accepted vertices)");
  desc.add<double>("fractionSumPt2", 0.3)->setComment("threshold on sumPt2 fraction of the leading vertex");
  desc.add<double>("minSumPt2", 0.)->setComment("min sumPt2");
  edm::ParameterSetDescription PVcomparerPSet;
  PVcomparerPSet.add<double>("track_pt_min", 1.0)->setComment("min track p_T");
  PVcomparerPSet.add<double>("track_pt_max", 10.0)->setComment("max track p_T");
  PVcomparerPSet.add<double>("track_chi2_max", 99999.)->setComment("max track chi2");
  PVcomparerPSet.add<double>("track_prob_min", -1.)->setComment("min track prob");
  desc.add<edm::ParameterSetDescription>("PVcomparer", PVcomparerPSet)
      ->setComment("from RecoPixelVertexing/PixelVertexFinding/python/PVClusterComparer_cfi.py");
  descriptions.add("hltPixelVertexCollectionTrimmer", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(PixelVertexCollectionTrimmer);
