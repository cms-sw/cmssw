#include "HeavyIonsAnalysis/PhotonAnalysis/src/towerIsoCalculator.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"


using namespace edm;
using namespace reco;

#define PI 3.141592653589793238462643383279502884197169399375105820974945


towerIsoCalculator::towerIsoCalculator(const edm::Event &iEvent, const edm::EventSetup &iSetup, const edm::InputTag &towerCandidateLabel_, const edm::InputTag &towerVoroniBkgLabel_, const edm::InputTag &vtxLabel_)
{

  using namespace edm;
  using namespace reco;

  iEvent.getByLabel( towerCandidateLabel_,towers);
  iEvent.getByLabel( towerCandidateLabel_,candidatesView_);

  
  // voronoi background
  iEvent.getByLabel(towerVoroniBkgLabel_,towerVoronoiBkg);

  // vertex
  iEvent.getByLabel(vtxLabel_,vtxs);
  int greatestvtx = 0;
  int nVertex = vtxs->size();

  for (unsigned int i = 0 ; i< vtxs->size(); ++i){
    unsigned int daughter = (*vtxs)[i].tracksSize();
    if( daughter > (*vtxs)[greatestvtx].tracksSize()) greatestvtx = i;
  }
  if(nVertex<=0){
    vtx = reco::Vertex::Point(0,0,0);
  }
  else 
    vtx =  (*vtxs)[greatestvtx].position();

  // geometry
  edm::ESHandle<CaloGeometry> pGeo;
  iSetup.get<CaloGeometryRecord>().get(pGeo);
  geo = pGeo.product();
  
} 



double towerIsoCalculator::getEt(const DetId &id, double energy, reco::Vertex::Point vtx){
  const GlobalPoint& pos=geo->getPosition(id);
  double et = energy*sin(pos.theta());
  return et;
}

double towerIsoCalculator::getEta(const DetId &id, reco::Vertex::Point vtx){
  const GlobalPoint& pos=geo->getPosition(id);
  double et = pos.eta();
  return et;
}

double towerIsoCalculator::getPhi(const DetId &id, reco::Vertex::Point vtx){
  const GlobalPoint& pos=geo->getPosition(id);
  double et = pos.phi();
  return et;
}

double towerIsoCalculator::getTowerIso(const reco::Photon& photon, double r1, double r2, double jWidth, double threshold)
{
  using namespace edm;
  using namespace reco;

  double photonEta  = photon.eta();
  double photonPhi  = photon.phi();
  double TotalEt = 0;

  for(unsigned int i = 0; i < towers->size(); ++i){
    const CaloTower & hit= (*towers)[i];

    double towerEt = hit.p4(vtx).Et();
    double towerEta = hit.p4(vtx).Eta();
    double towerPhi = hit.p4(vtx).Phi();


    if ( towerEt <threshold) continue;

    double dEta = fabs( photonEta - towerEta);
    double dPhi = towerPhi - photonPhi; 
    while ( fabs(dPhi) > PI) {
      if ( dPhi > PI )  dPhi = dPhi - 2.*PI;
      if ( dPhi < PI*(-1.) )  dPhi = dPhi + 2.*PI;
    }
    double dR = sqrt(dEta*dEta+dPhi*dPhi);
     
    // Jurassic Cone /////
    if ( dR > r1 ) continue;
    if ( dR < r2 ) continue;
    if ( fabs(dEta) <  jWidth)  continue;
    TotalEt += towerEt;
  }
  
  return TotalEt;
}

double towerIsoCalculator::getVsTowerIso(const reco::Photon& photon, double r1, double r2, double jWidth, double threshold, bool isVsCorrected)
{
  using namespace edm;
  using namespace reco;

  double photonEta  = photon.eta();
  double photonPhi  = photon.phi();
  double TotalEt = 0;

  for(unsigned int i = 0; i < towers->size(); ++i){
    const CaloTower & hit= (*towers)[i];

    //   double towerEt = hit.p4(vtx).Et();
    double towerEta = hit.p4(vtx).Eta();
    double towerPhi = hit.p4(vtx).Phi();


    //  if ( towerEt <threshold) continue;

    double dEta = fabs( photonEta - towerEta);
    double dPhi = towerPhi - photonPhi;
    while ( fabs(dPhi) > PI) {
      if ( dPhi > PI )  dPhi = dPhi - 2.*PI;
      if ( dPhi < PI*(-1.) )  dPhi = dPhi + 2.*PI;
    }
    double dR = sqrt(dEta*dEta+dPhi*dPhi);
    
    // Jurassic Cone /////
    if ( dR > r1 ) continue;
    if ( dR < r2 ) continue;
    if ( fabs(dEta) <  jWidth)  continue;

    // voronoi background
    reco::CandidateViewRef ref(candidatesView_,i);
    const reco::VoronoiBackground& voronoi = (*towerVoronoiBkg)[ref];

    if ( isVsCorrected)  TotalEt = TotalEt + voronoi.pt(); 
    else               TotalEt = TotalEt + voronoi.pt_subtracted(); 
  }

  return TotalEt;
}
