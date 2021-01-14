//--------------------------------------------------------------------------------------------------
// $Id $
//
// SCEnergyCorrectorSemiParm
//
// Helper Class for applying regression-based energy corrections with optimized BDT implementation
//
// Authors: J.Bendavid
//--------------------------------------------------------------------------------------------------

#ifndef RecoEcal_EgammaClusterAlgos_SCEnergyCorrectorSemiParm_h
#define RecoEcal_EgammaClusterAlgos_SCEnergyCorrectorSemiParm_h

#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "CondFormats/GBRForest/interface/GBRForestD.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "CondFormats/DataRecord/interface/GBRDWrapperRcd.h"

class SCEnergyCorrectorSemiParm {
public:
  SCEnergyCorrectorSemiParm();
  ~SCEnergyCorrectorSemiParm();

  void setTokens(const edm::ParameterSet &iConfig, edm::ConsumesCollector &cc);

  std::pair<double, double> getCorrections(const reco::SuperCluster &sc) const;
  void modifyObject(reco::SuperCluster &sc);

  void setEventSetup(const edm::EventSetup &es);
  void setEvent(const edm::Event &e);

protected:
  const GBRForestD *foresteb_;
  const GBRForestD *forestee_;
  const GBRForestD *forestsigmaeb_;
  const GBRForestD *forestsigmaee_;

  edm::ESHandle<CaloTopology> calotopo_;
  edm::ESHandle<CaloGeometry> calogeom_;

  edm::ESGetToken<CaloTopology, CaloTopologyRecord> tokenCaloTopo_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> tokenCaloGeom_;
  edm::ESGetToken<GBRForestD, GBRDWrapperRcd> tokenRegressionKeyEB_;
  edm::ESGetToken<GBRForestD, GBRDWrapperRcd> tokenUncertaintyKeyEB_;
  edm::ESGetToken<GBRForestD, GBRDWrapperRcd> tokenRegressionKeyEE_;
  edm::ESGetToken<GBRForestD, GBRDWrapperRcd> tokenUncertaintyKeyEE_;

  edm::EDGetTokenT<EcalRecHitCollection> tokenEBRecHits_;
  edm::EDGetTokenT<EcalRecHitCollection> tokenEERecHits_;
  edm::EDGetTokenT<reco::VertexCollection> tokenVertices_;

  edm::Handle<reco::VertexCollection> vertices_;
  edm::Handle<EcalRecHitCollection> rechitsEB_;
  edm::Handle<EcalRecHitCollection> rechitsEE_;

  edm::InputTag ecalHitsEBInputTag_;
  edm::InputTag ecalHitsEEInputTag_;
  edm::InputTag vertexInputTag_;

  std::string regressionKeyEB_;
  std::string uncertaintyKeyEB_;
  std::string regressionKeyEE_;
  std::string uncertaintyKeyEE_;

private:
  bool isHLT_;
  bool applySigmaIetaIphiBug_;  //there was a bug in sigmaIetaIphi for the 74X application
  int nHitsAboveThreshold_;
  float eThreshold_;
};
#endif
