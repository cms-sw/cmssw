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


  //edm::Handle<reco::PFCandidateCollection> pfCandidates;
  edm::Handle<reco::CandidateView> candidatesView;
  //iEvent.getByLabel(pfCandidateLabel_,pfCandidates);
  iEvent.getByLabel(pfCandidateLabel_,candidatesView);
  // pfCandidateColl = pfCandidates.product();

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
  //edm::Handle<reco::VoronoiMap> pfVoronoiBkg;
  iEvent.getByLabel(pfVoroniBkgLabel_,pfVoronoiBkg);

}

pfIsoCalculator::pfIsoCalculator(const edm::Event &iEvent, const edm::EventSetup &iSetup, const edm::EDGetTokenT<edm::View<reco::PFCandidate> > pfCandidates_, const edm::EDGetTokenT<edm::ValueMap<reco::VoronoiBackground> > pfVoronoiBkg_, const math::XYZPoint& pv)
{

  using namespace edm;
  using namespace reco;


  //edm::Handle<reco::PFCandidateCollection> pfCandidates;
  //edm::Handle<reco::CandidateView> candidatesView;
  //iEvent.getByToken(pfCandidates_,pfCandidates);
  iEvent.getByToken(pfCandidates_,candidatesView);
  //pfCandidateColl = pfCandidates.product();
  //pfCandidateColl = candidatesView.product();

  // vertex
  vtx_ = pv;

  // voronoi background
  //  edm::Handle<reco::VoronoiMap> pfVoronoiBkg;
  if(!(pfVoronoiBkg_.isUninitialized()))
    iEvent.getByToken(pfVoronoiBkg_,pfVoronoiBkg);

}



double pfIsoCalculator::getPfIso(const reco::Photon& photon, int pfId, double r1, double r2, double jWidth, double threshold)
{
  using namespace edm;
  using namespace reco;

  double photonEta  = photon.eta();
  double photonPhi  = photon.phi();
  double TotalEt = 0;

  // if (!pfCandidateColl)
  //return -999;

  //for(unsigned icand=0;icand<pfCandidateColl->size(); icand++) {
  for (edm::View<reco::PFCandidate>::const_iterator pf = candidatesView->begin(); pf != candidatesView->end(); ++pf) {

    //const reco::PFCandidate pfCandidate = pfCandidateColl->at(icand);
    if ( pf->particleId() != pfId )   continue;
    double pfEta = pf->eta();
    double pfPhi = pf->phi();

    double dEta = fabs( photonEta - pfEta);
    double dPhi = pfPhi - photonPhi;
    while ( fabs(dPhi) > PI) {
      if ( dPhi > PI )  dPhi = dPhi - 2.*PI;
      if ( dPhi < PI*(-1.) )  dPhi = dPhi + 2.*PI;
    }
    double dR = sqrt(dEta*dEta+dPhi*dPhi);
    double thePt = pf->pt();


    // Remove the photon itself
    if ( pf->superClusterRef() == photon.superCluster() ) continue;

    if( pf->particleId()==reco::PFCandidate::h){
      float dz = fabs( pf->vz() - vtx_.z());
      if (dz > 0.2) continue;
      double dxy = ( -( pf->vx() - vtx_.x())*pf->py() + (pf->vy() - vtx_.y())*pf->px()) / pf->pt();
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

  //if (!pfCandidateColl)
  //return -999;

  //for(unsigned icand=0;icand<pfCandidateColl->size(); icand++) {
  //for(vector<reco::PFCandidate>::const_iterator pf = pfCandidateColl->begin(); pf != pfCandidateColl->end(); pf++)
  //const reco::PFCandidate pfCandidate = pfCandidateColl->at(icand);
  //const reco::PFCandidate pfCandidate = *pf;
  for (edm::View<reco::PFCandidate>::const_iterator pf = candidatesView->begin(); pf != candidatesView->end(); ++pf) {
    if ( pf->particleId() != pfId )   continue;

    double pfEta = pf->eta();
    double pfPhi = pf->phi();

    double dEta = fabs( photonEta - pfEta);
    double dPhi = pfPhi - photonPhi;
    while ( fabs(dPhi) > PI) {
      if ( dPhi > PI )  dPhi = dPhi - 2.*PI;
      if ( dPhi < PI*(-1.) )  dPhi = dPhi + 2.*PI;
    }

    double dR = sqrt(dEta*dEta+dPhi*dPhi);
    double thePt = pf->pt();
    // Jurassic Cone //
    if ( dR > r1 ) continue;
    if ( dR < r2 ) continue;
    if ( fabs(dEta) <  jWidth)  continue;
    if (thePt<threshold) continue;


    // voronoi background
    //reco::CandidateViewRef ref(candidatesView,icand);
    //edm::RefToBase<reco::PFCandidate> ref = pfCandidateColl->refAt(icand);
    unsigned int idx = pf - candidatesView->begin();
    edm::RefToBase<reco::PFCandidate> ref = candidatesView->refAt(idx);

    const reco::VoronoiBackground& voronoi = (*pfVoronoiBkg)[ref];

    if ( isVsCorrected)  TotalEt = TotalEt + voronoi.pt();
    else               TotalEt = TotalEt + voronoi.pt_subtracted();

  }

  return TotalEt;
}

double pfIsoCalculator::getPfIso(const reco::GsfElectron& ele, int pfId, double r1, double r2, double threshold)
{

  using namespace edm;
  using namespace reco;

  double eleEta = ele.eta();
  double elePhi = ele.phi();
  double TotalEt = 0.;

  for (edm::View<reco::PFCandidate>::const_iterator pf = candidatesView->begin(); pf != candidatesView->end(); ++pf) {
    if ( pf->particleId() != pfId )   continue;
    double pfEta = pf->eta();
    double pfPhi = pf->phi();

    double dEta = fabs( eleEta - pfEta);
    double dPhi = pfPhi - elePhi;
    while ( fabs(dPhi) > PI) {
      if ( dPhi > PI )  dPhi = dPhi - 2.*PI;
      if ( dPhi < PI*(-1.) )  dPhi = dPhi + 2.*PI;
    }
    double dR = sqrt(dEta*dEta+dPhi*dPhi);
    double thePt = pf->pt();

    // remove electron itself
    if (pf->particleId() == reco::PFCandidate::e) continue;

    if (pf->particleId() == reco::PFCandidate::h) {
      float dz = fabs(pf->vz() - vtx_.z());
      if (dz > 0.2) continue;
      double dxy = ( -(pf->vx() - vtx_.x())*pf->py() + (pf->vy() - vtx_.y())*pf->px()) / pf->pt();
      if ( fabs(dxy) > 0.1) continue;
    }

    //inside the cone size
    if (dR > r1) continue;
    if (dR < r2) continue;
    if (thePt < threshold) continue;
    TotalEt += thePt;

  }

  return TotalEt;

}
