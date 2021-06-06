#include "RecoEgamma/EgammaTools/interface/BaselinePFSCRegression.h"
#include "RecoEgamma/EgammaTools/interface/EcalRegressionData.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"

#include "TVector2.h"
#include "DataFormats/Math/interface/deltaR.h"

void BaselinePFSCRegression::update(const edm::EventSetup& es) {
  const CaloTopologyRecord& topofrom_es = es.get<CaloTopologyRecord>();
  if (!topo_record || topofrom_es.cacheIdentifier() != topo_record->cacheIdentifier()) {
    topo_record = &topofrom_es;
    topo_record->get(calotopo);
  }
  const CaloGeometryRecord& geomfrom_es = es.get<CaloGeometryRecord>();
  if (!geom_record || geomfrom_es.cacheIdentifier() != geom_record->cacheIdentifier()) {
    geom_record = &geomfrom_es;
    geom_record->get(calogeom);
  }
}

void BaselinePFSCRegression::set(const reco::SuperCluster& sc, std::vector<float>& vars) const {
  EcalRegressionData regData;
  regData.fill(sc, rechitsEB, rechitsEE, calogeom.product(), calotopo.product(), vertices);
  regData.fillVec(vars);

  //solely to reproduce old exception behaviour, unnessessary although it likely is
  if (sc.seed()->hitsAndFractions().at(0).first.subdetId() != EcalBarrel &&
      sc.seed()->hitsAndFractions().at(0).first.subdetId() != EcalEndcap) {
    throw cms::Exception("PFECALSuperClusterProducer::calculateRegressedEnergy")
        << "Supercluster seed is either EB nor EE!" << std::endl;
  }
}

void BaselinePFSCRegression::setTokens(const edm::ParameterSet& ps, edm::ConsumesCollector&& cc) {
  inputTagEBRecHits_ = cc.consumes(ps.getParameter<edm::InputTag>("ecalRecHitsEB"));
  inputTagEERecHits_ = cc.consumes(ps.getParameter<edm::InputTag>("ecalRecHitsEE"));
  inputTagVertices_ = cc.consumes(ps.getParameter<edm::InputTag>("vertexCollection"));
}

void BaselinePFSCRegression::setEvent(const edm::Event& ev) {
  rechitsEB = &ev.get(inputTagEBRecHits_);
  rechitsEE = &ev.get(inputTagEERecHits_);
  vertices = &ev.get(inputTagVertices_);
}
