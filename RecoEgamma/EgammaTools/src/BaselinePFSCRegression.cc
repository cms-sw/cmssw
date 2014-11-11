#include "RecoEgamma/EgammaTools/interface/BaselinePFSCRegression.h"
#include "RecoEgamma/EgammaTools/interface/EcalRegressionData.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"

#include "TVector2.h"
#include "DataFormats/Math/interface/deltaR.h"

void BaselinePFSCRegression::update(const edm::EventSetup& es) {
  const CaloTopologyRecord& topofrom_es = es.get<CaloTopologyRecord>();
  if( !topo_record ||
      topofrom_es.cacheIdentifier() != topo_record->cacheIdentifier() ) {
    topo_record = &topofrom_es;
    topo_record->get(calotopo);
  }
  const CaloGeometryRecord& geomfrom_es = es.get<CaloGeometryRecord>();
  if( !geom_record ||
      geomfrom_es.cacheIdentifier() != geom_record->cacheIdentifier() ) {
    geom_record = &geomfrom_es;
    geom_record->get(calogeom);
  }
}

void BaselinePFSCRegression::set(const reco::SuperCluster& sc,
				 std::vector<float>& vars     ) const {
  EcalRegressionData regData;
  regData.fill(sc,rechitsEB.product(),rechitsEE.product(),calogeom.product(),calotopo.product(),vertices.product());
  regData.fillVec(vars);

  //solely to reproduce old exception behaviour, unnessessary although it likely is
  if( sc.seed()->hitsAndFractions().at(0).first.subdetId()!=EcalBarrel &&
      sc.seed()->hitsAndFractions().at(0).first.subdetId()!=EcalEndcap){
   throw cms::Exception("PFECALSuperClusterProducer::calculateRegressedEnergy")
     << "Supercluster seed is either EB nor EE!" << std::endl;
  }

 
}

void BaselinePFSCRegression::
setTokens(const edm::ParameterSet& ps, edm::ConsumesCollector&& cc) {
  const edm::InputTag rceb = ps.getParameter<edm::InputTag>("ecalRecHitsEB");
  const edm::InputTag rcee = ps.getParameter<edm::InputTag>("ecalRecHitsEE");
  const edm::InputTag vtx = ps.getParameter<edm::InputTag>("vertexCollection");
  inputTagEBRecHits_      = cc.consumes<EcalRecHitCollection>(rceb);
  inputTagEERecHits_      = cc.consumes<EcalRecHitCollection>(rcee);
  inputTagVertices_       = cc.consumes<reco::VertexCollection>(vtx);
}

void BaselinePFSCRegression::setEvent(const edm::Event& ev) {
  ev.getByToken(inputTagEBRecHits_,rechitsEB);
  ev.getByToken(inputTagEERecHits_,rechitsEE);
  ev.getByToken(inputTagVertices_,vertices);
}
