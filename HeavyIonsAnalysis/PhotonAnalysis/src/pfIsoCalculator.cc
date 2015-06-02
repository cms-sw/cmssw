#include "HeavyIonsAnalysis/PhotonAnalysis/src/pfIsoCalculator.h"

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


pfIsoCalculator::pfIsoCalculator(const edm::Event &iEvent, const edm::EventSetup &iSetup, const edm::InputTag &pfCandidateLabel_, const edm::InputTag &pfVoroniBkgLabel_,  const edm::InputTag &vtxLabel_ ) 
{

  using namespace edm;
  using namespace reco;


  edm::Handle<reco::PFCandidateCollection> pfCandidates;
  //  edm::Handle<reco::CandidateView> candidatesView;
  iEvent.getByLabel(pfCandidateLabel_,pfCandidates);
  iEvent.getByLabel(pfCandidateLabel_,candidatesView);
  pfCandidateColl = pfCandidates.product();
  
  // vertex
  iEvent.getByLabel(vtxLabel_,vtxs);
  int greatestvtx = 0;
  int nVertex = vtxs->size();
  for (unsigned int i = 0 ; i< vtxs->size(); ++i){
    unsigned int daughter = (*vtxs)[i].tracksSize();
    if( daughter > (*vtxs)[greatestvtx].tracksSize()) greatestvtx = i;
  }
  if(nVertex<=0){
    vtx_ = reco::Vertex::Point(0,0,0);
  }
  else
    vtx_ =  (*vtxs)[greatestvtx].position();


  // voronoi background
  //  edm::Handle<reco::VoronoiMap> pfVoronoiBkg;
  iEvent.getByLabel(pfVoroniBkgLabel_,pfVoronoiBkg);

} 


double pfIsoCalculator::getPfIso(const reco::Photon& photon, int pfId, double r1, double r2, double jWidth, double threshold)
{
  using namespace edm;
  using namespace reco;

  double photonEta  = photon.eta();
  double photonPhi  = photon.phi();
  double TotalEt = 0;

  if (!pfCandidateColl)  
    return -999;
 
  for(unsigned icand=0;icand<pfCandidateColl->size(); icand++) {
    
    const reco::PFCandidate pfCandidate = pfCandidateColl->at(icand);
    if ( pfCandidate.particleId() != pfId )   continue;
    double pfEta = pfCandidate.eta();
    double pfPhi = pfCandidate.phi();

    double dEta = fabs( photonEta - pfEta);
    double dPhi = pfPhi - photonPhi; 
    while ( fabs(dPhi) > PI) {
      if ( dPhi > PI )  dPhi = dPhi - 2.*PI;
      if ( dPhi < PI*(-1.) )  dPhi = dPhi + 2.*PI;
    }
    double dR = sqrt(dEta*dEta+dPhi*dPhi);
    double thePt = pfCandidate.pt();

    
    // Remove the photon itself 
    if ( pfCandidate.superClusterRef() == photon.superCluster() ) continue;
    
    if( pfCandidate.particleId()==reco::PFCandidate::h){
      float dz = fabs( pfCandidate.vz() - vtx_.z());
      if (dz > 0.2) continue;
      double dxy = ( -( pfCandidate.vx() - vtx_.x())*pfCandidate.py() + (pfCandidate.vy() - vtx_.y())*pfCandidate.px()) / pfCandidate.pt();
      if(fabs(dxy) > 0.1) continue;
    }


    
    
    // Jurassic Cone /////
    if ( dR > r1 ) continue;
    if ( dR < r2 ) continue;
    if ( fabs(dEta) <  jWidth)  continue;
    if (thePt<threshold) continue;
    TotalEt += thePt;
  }
  
  return TotalEt;
}

double pfIsoCalculator::getVsPfIso(const reco::Photon& photon, int pfId, double r1, double r2, double jWidth, double threshold, bool isVsCorrected)
{
  using namespace edm;
  using namespace reco;

  double photonEta  = photon.eta();
  double photonPhi  = photon.phi();
  double TotalEt = 0;

  if (!pfCandidateColl)
    return -999;

  for(unsigned icand=0;icand<pfCandidateColl->size(); icand++) {

    const reco::PFCandidate pfCandidate = pfCandidateColl->at(icand);
    if ( pfCandidate.particleId() != pfId )   continue;
    
    double pfEta = pfCandidate.eta();
    double pfPhi = pfCandidate.phi();

    double dEta = fabs( photonEta - pfEta);
    double dPhi = pfPhi - photonPhi;
    while ( fabs(dPhi) > PI) {
      if ( dPhi > PI )  dPhi = dPhi - 2.*PI;
      if ( dPhi < PI*(-1.) )  dPhi = dPhi + 2.*PI;
    }
    
    double dR = sqrt(dEta*dEta+dPhi*dPhi);
    double thePt = pfCandidate.pt();
    // Jurassic Cone //
    if ( dR > r1 ) continue;
    if ( dR < r2 ) continue;
    if ( fabs(dEta) <  jWidth)  continue;
    if (thePt<threshold) continue;


    // voronoi background
    reco::CandidateViewRef ref(candidatesView,icand);
    const reco::VoronoiBackground& voronoi = (*pfVoronoiBkg)[ref];

    if ( isVsCorrected)  TotalEt = TotalEt + voronoi.pt();
    else               TotalEt = TotalEt + voronoi.pt_subtracted();
    
  }
  
  return TotalEt;
}
