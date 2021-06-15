//--------------------------------------------------------------------------------------------------
// $Id $
//
// SCEnergyCorrectorSemiParm
//
// Helper Class for applying regression-based energy corrections with optimized BDT implementation
//
// Original Author: J.Bendavid
//
// Refactored, modernised and extended to HGCAL by S. Harper (RAL/CERN)
// with input from S. Bhattacharya (DESY)
//--------------------------------------------------------------------------------------------------

#ifndef RecoEcal_EgammaClusterAlgos_SCEnergyCorrectorSemiParm_h
#define RecoEcal_EgammaClusterAlgos_SCEnergyCorrectorSemiParm_h

#include "FWCore/Framework/interface/Event.h"
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
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "CondFormats/GBRForest/interface/GBRForestD.h"
#include "CondFormats/DataRecord/interface/GBRDWrapperRcd.h"
#include "RecoEgamma/EgammaTools/interface/EgammaBDTOutputTransformer.h"
#include "RecoEgamma/EgammaTools/interface/HGCalShowerShapeHelper.h"

class SCEnergyCorrectorSemiParm {
public:
  SCEnergyCorrectorSemiParm();
  //if you want override the default on where conditions are consumed, you need to use
  //the other constructor and then call setTokens approprately
  SCEnergyCorrectorSemiParm(const edm::ParameterSet& iConfig, edm::ConsumesCollector cc);

  static void fillPSetDescription(edm::ParameterSetDescription& desc);
  static edm::ParameterSetDescription makePSetDescription();

  template <edm::Transition tr = edm::Transition::BeginLuminosityBlock>
  void setTokens(const edm::ParameterSet& iConfig, edm::ConsumesCollector cc);

  void setEventSetup(const edm::EventSetup& es);
  void setEvent(const edm::Event& e);

  std::pair<double, double> getCorrections(const reco::SuperCluster& sc) const;
  void modifyObject(reco::SuperCluster& sc) const;

  std::vector<float> getRegData(const reco::SuperCluster& sc) const;

private:
  class RegParam {
  public:
    RegParam(std::string meanKey = "",
             float meanLow = 0,
             float meanHigh = 0,
             std::string sigmaKey = "",
             float sigmaLow = 0,
             float sigmaHigh = 0)
        : meanKey_(std::move(meanKey)),
          sigmaKey_(std::move(sigmaKey)),
          meanOutTrans_(meanLow, meanHigh),
          sigmaOutTrans_(sigmaLow, sigmaHigh) {}
    RegParam(edm::ConsumesCollector cc,
             std::string meanKey = "",
             float meanLow = 0,
             float meanHigh = 0,
             std::string sigmaKey = "",
             float sigmaLow = 0,
             float sigmaHigh = 0)
        : RegParam(meanKey, meanLow, meanHigh, sigmaKey, sigmaLow, sigmaHigh) {
      setTokens(cc);
    }
    template <edm::Transition esTransition = edm::Transition::BeginLuminosityBlock>
    void setTokens(edm::ConsumesCollector cc);
    void setForests(const edm::EventSetup& setup);

    double mean(const std::vector<float>& data) const;
    double sigma(const std::vector<float>& data) const;

  private:
    std::string meanKey_;
    std::string sigmaKey_;
    EgammaBDTOutputTransformer meanOutTrans_;
    EgammaBDTOutputTransformer sigmaOutTrans_;
    const GBRForestD* meanForest_;
    const GBRForestD* sigmaForest_;
    edm::ESGetToken<GBRForestD, GBRDWrapperRcd> meanForestToken_;
    edm::ESGetToken<GBRForestD, GBRDWrapperRcd> sigmaForestToken_;
  };

  //returns barrel for ecal barrel, otherwise returns endcap
  const RegParam& getRegParam(const DetId& detId) const {
    return detId.det() == DetId::Ecal && detId.subdetId() == EcalBarrel ? regParamBarrel_ : regParamEndcap_;
  }

  std::vector<float> getRegDataECALV1(const reco::SuperCluster& sc) const;
  std::vector<float> getRegDataECALHLTV1(const reco::SuperCluster& sc) const;
  std::vector<float> getRegDataHGCALV1(const reco::SuperCluster& sc) const;
  std::vector<float> getRegDataHGCALHLTV1(const reco::SuperCluster& sc) const;

  //barrel = always ecal barrel, endcap may be ECAL or HGCAL
  RegParam regParamBarrel_;
  RegParam regParamEndcap_;

  const CaloTopology* caloTopo_;
  const CaloGeometry* caloGeom_;
  edm::ESGetToken<CaloTopology, CaloTopologyRecord> caloTopoToken_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeomToken_;

  edm::EDGetTokenT<EcalRecHitCollection> tokenEBRecHits_;
  edm::EDGetTokenT<EcalRecHitCollection> tokenEERecHits_;
  edm::EDGetTokenT<reco::PFRecHitCollection> tokenHgcalRecHits_;
  edm::EDGetTokenT<reco::VertexCollection> tokenVertices_;

  edm::Handle<EcalRecHitCollection> recHitsEB_;
  edm::Handle<EcalRecHitCollection> recHitsEE_;
  edm::Handle<reco::PFRecHitCollection> recHitsHgcal_;
  edm::Handle<reco::VertexCollection> vertices_;

  bool isHLT_;
  bool isPhaseII_;
  bool applySigmaIetaIphiBug_;  //there was a bug in sigmaIetaIphi for the 74X application
  int nHitsAboveThresholdEB_;
  int nHitsAboveThresholdEE_;
  int nHitsAboveThresholdHG_;
  float hitsEnergyThreshold_;
  float hgcalCylinderR_;
  HGCalShowerShapeHelper hgcalShowerShapes_;
};

template <edm::Transition esTransition>
void SCEnergyCorrectorSemiParm::RegParam::setTokens(edm::ConsumesCollector cc) {
  meanForestToken_ = cc.esConsumes<GBRForestD, GBRDWrapperRcd, esTransition>(edm::ESInputTag("", meanKey_));
  sigmaForestToken_ = cc.esConsumes<GBRForestD, GBRDWrapperRcd, esTransition>(edm::ESInputTag("", sigmaKey_));
}

template <edm::Transition esTransition>
void SCEnergyCorrectorSemiParm::setTokens(const edm::ParameterSet& iConfig, edm::ConsumesCollector cc) {
  isHLT_ = iConfig.getParameter<bool>("isHLT");
  isPhaseII_ = iConfig.getParameter<bool>("isPhaseII");
  applySigmaIetaIphiBug_ = iConfig.getParameter<bool>("applySigmaIetaIphiBug");
  tokenEBRecHits_ = cc.consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("ecalRecHitsEB"));
  if (not isPhaseII_) {
    tokenEERecHits_ = cc.consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("ecalRecHitsEE"));
  } else {
    tokenHgcalRecHits_ = cc.consumes<reco::PFRecHitCollection>(iConfig.getParameter<edm::InputTag>("hgcalRecHits"));
    hgcalCylinderR_ = iConfig.getParameter<double>("hgcalCylinderR");
    hgcalShowerShapes_.setTokens<esTransition>(cc);
  }
  caloGeomToken_ = cc.esConsumes<CaloGeometry, CaloGeometryRecord, esTransition>();
  caloTopoToken_ = cc.esConsumes<CaloTopology, CaloTopologyRecord, esTransition>();

  regParamBarrel_ = RegParam(iConfig.getParameter<std::string>("regressionKeyEB"),
                             iConfig.getParameter<double>("regressionMinEB"),
                             iConfig.getParameter<double>("regressionMaxEB"),
                             iConfig.getParameter<std::string>("uncertaintyKeyEB"),
                             iConfig.getParameter<double>("uncertaintyMinEB"),
                             iConfig.getParameter<double>("uncertaintyMaxEB"));
  regParamBarrel_.setTokens<esTransition>(cc);
  regParamEndcap_ = RegParam(iConfig.getParameter<std::string>("regressionKeyEE"),
                             iConfig.getParameter<double>("regressionMinEE"),
                             iConfig.getParameter<double>("regressionMaxEE"),
                             iConfig.getParameter<std::string>("uncertaintyKeyEE"),
                             iConfig.getParameter<double>("uncertaintyMinEE"),
                             iConfig.getParameter<double>("uncertaintyMaxEE"));
  regParamEndcap_.setTokens<esTransition>(cc);
  hitsEnergyThreshold_ = iConfig.getParameter<double>("eRecHitThreshold");
  if (not isHLT_) {
    tokenVertices_ = cc.consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertexCollection"));
  }
}
#endif
