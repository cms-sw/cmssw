//--------------------------------------------------------------------------------------------------
//
// SCEnergyCorrectorDRN
//
// Helper Class for applying regression-based energy corrections with DRN implimentation
//
// Based on RecoEcal/EgammaClusterAlgos/SCEnergyCorrectorSemiParm
//
// Author: Simon Rothman (MIT, UMN)
//
//--------------------------------------------------------------------------------------------------

#ifndef RecoEcal_EgammaClusterAlgos_SCEnergyCorrectorDRN_h
#define RecoEcal_EgammaClusterAlgos_SCEnergyCorrectorDRN_h

#include "HeterogeneousCore/SonicTriton/interface/TritonData.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/Records/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "CondFormats/GBRForest/interface/GBRForestD.h"
#include "CondFormats/DataRecord/interface/GBRDWrapperRcd.h"

#include "RecoEgamma/EgammaTools/interface/EgammaBDTOutputTransformer.h"
#include "RecoEgamma/EgammaTools/interface/HGCalShowerShapeHelper.h"

#include <sstream>
#include <string>
#include <vector>
#include <random>

class SCEnergyCorrectorDRN {
public:
  SCEnergyCorrectorDRN();
  //if you want override the default on where conditions are consumed, you need to use
  //the other constructor and then call setTokens approprately
  SCEnergyCorrectorDRN(const edm::ParameterSet& iConfig, edm::ConsumesCollector cc);

  static void fillPSetDescription(edm::ParameterSetDescription& desc);
  static edm::ParameterSetDescription makePSetDescription();

  template <edm::Transition tr = edm::Transition::BeginLuminosityBlock>
  void setTokens(const edm::ParameterSet& iConfig, edm::ConsumesCollector cc);

  void setEventSetup(const edm::EventSetup& es);
  void setEvent(const edm::Event& e);

  void makeInput(const edm::Event& iEvent, TritonInputMap& iInput, const reco::SuperClusterCollection& inputSCs) const;
  TritonOutput<float> getOutput(const TritonOutputMap& iOutput);

private:
  const CaloTopology* caloTopo_;
  const CaloGeometry* caloGeom_;
  edm::ESGetToken<CaloTopology, CaloTopologyRecord> caloTopoToken_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeomToken_;

  edm::EDGetTokenT<EcalRecHitCollection> tokenEBRecHits_;
  edm::EDGetTokenT<EcalRecHitCollection> tokenEERecHits_;
  edm::EDGetTokenT<double> rhoToken_;

  edm::Handle<EcalRecHitCollection> recHitsEB_;
  edm::Handle<EcalRecHitCollection> recHitsEE_;

  edm::Handle<double> rhoHandle_;
};

template <edm::Transition esTransition>
void SCEnergyCorrectorDRN::setTokens(const edm::ParameterSet& iConfig, edm::ConsumesCollector cc) {
  tokenEBRecHits_ = cc.consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("ecalRecHitsEB"));
  tokenEERecHits_ = cc.consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("ecalRecHitsEE"));
  caloGeomToken_ = cc.esConsumes<CaloGeometry, CaloGeometryRecord, esTransition>();
  caloTopoToken_ = cc.esConsumes<CaloTopology, CaloTopologyRecord, esTransition>();
  rhoToken_ = cc.consumes<double>(iConfig.getParameter<edm::InputTag>("rhoFastJet"));
}
#endif
