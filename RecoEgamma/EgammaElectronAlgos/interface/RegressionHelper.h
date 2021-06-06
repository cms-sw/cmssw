#ifndef __EgammaElectronAlgos_RegressionHelper_h__
#define __EgammaElectronAlgos_RegressionHelper_h__

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "CondFormats/GBRForest/interface/GBRForest.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

#include "Geometry/Records/interface/CaloTopologyRecord.h"
#include "CondFormats/DataRecord/interface/GBRWrapperRcd.h"

class RegressionHelper {
public:
  struct Configuration {
    // weight files for the regression
    std::vector<std::string> ecalRegressionWeightLabels;
    bool ecalWeightsFromDB;
    std::vector<std::string> ecalRegressionWeightFiles;
    std::vector<std::string> combinationRegressionWeightLabels;
    bool combinationWeightsFromDB;
    std::vector<std::string> combinationRegressionWeightFiles;
  };

  struct ESGetTokens {
    ESGetTokens(Configuration const& cfg, bool useEcalReg, bool useCombinationReg, edm::ConsumesCollector& cc);

    edm::ESGetToken<CaloTopology, CaloTopologyRecord> caloTopology;
    edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeometry;
    edm::ESGetToken<GBRForest, GBRWrapperRcd> ecalRegBarrel;
    edm::ESGetToken<GBRForest, GBRWrapperRcd> ecalRegEndcap;
    edm::ESGetToken<GBRForest, GBRWrapperRcd> ecalRegErrorBarrel;
    edm::ESGetToken<GBRForest, GBRWrapperRcd> ecalRegErrorEndcap;
    edm::ESGetToken<GBRForest, GBRWrapperRcd> combinationReg;
  };

  RegressionHelper(Configuration const&, bool useEcalReg, bool useCombinationReg, edm::ConsumesCollector& cc);
  void checkSetup(const edm::EventSetup&);
  void applyEcalRegression(reco::GsfElectron& electron,
                           const reco::VertexCollection& vertices,
                           const EcalRecHitCollection& rechitsEB,
                           const EcalRecHitCollection& rechitsEE) const;

  void applyCombinationRegression(reco::GsfElectron& ele) const;

private:
  void getEcalRegression(const reco::SuperCluster& sc,
                         const reco::VertexCollection& vertices,
                         const EcalRecHitCollection& rechitsEB,
                         const EcalRecHitCollection& rechitsEE,
                         double& energyFactor,
                         double& errorFactor) const;

private:
  const Configuration cfg_;
  const ESGetTokens esGetTokens_;

  const CaloTopology* caloTopology_;
  const CaloGeometry* caloGeometry_;
  bool ecalRegressionInitialized_ = false;
  bool combinationRegressionInitialized_ = false;

  const GBRForest* ecalRegBarrel_;
  const GBRForest* ecalRegEndcap_;
  const GBRForest* ecalRegErrorBarrel_;
  const GBRForest* ecalRegErrorEndcap_;
  const GBRForest* combinationReg_;
};

#endif
