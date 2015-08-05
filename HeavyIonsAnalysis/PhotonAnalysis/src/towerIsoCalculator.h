#ifndef towerIsoCalculator_h
#define towerIsoCalculator_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

//#include "RecoCaloTools/MetaCollections/interface/CaloRecHitMetaCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/HeavyIonEvent/interface/VoronoiBackground.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/EcalDetId/interface/EcalDetIdCollections.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/DetId/interface/DetId.h"


#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

class towerIsoCalculator
{

 public:

  towerIsoCalculator(const edm::Event &iEvent, const edm::EventSetup &iSetup, const edm::InputTag &towerCandidateLabel_, const edm::InputTag &towerVoroniBkgLabel_, const edm::InputTag &vtxLabel_) ;
  double getTowerIso (const reco::Photon& photon, double r1=0.4, double r2=0.06, double jWidth=0.04, double threshold=0);
  double getVsTowerIso(const reco::Photon& photon, double r1=0.4, double r2=0.06, double jWidth=0.04, double threshold=0, bool isVsCorrected=true);
  double       getEt(const DetId &id, double energy, reco::Vertex::Point vtx = reco::Vertex::Point(0,0,0));
  double       getEta(const DetId &id, reco::Vertex::Point vtx = reco::Vertex::Point(0,0,0));
  double       getPhi(const DetId &id, reco::Vertex::Point vtx = reco::Vertex::Point(0,0,0));
  reco::Vertex::Point getVtx(const edm::Event& ev);

 private:
  edm::Handle<CaloTowerCollection> towers;
  edm::Handle<reco::CandidateView> candidatesView_;
  edm::Handle<reco::VoronoiMap> towerVoronoiBkg;
  edm::Handle<reco::VertexCollection> vtxs;
  reco::Vertex::Point vtx;
  const CaloGeometry *geo;
};

#endif
