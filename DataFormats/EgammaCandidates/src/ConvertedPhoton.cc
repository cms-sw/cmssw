
#include "DataFormats/EgammaCandidates/interface/ConvertedPhoton.h"

using namespace reco;




ConvertedPhoton::ConvertedPhoton( const reco::SuperClusterRef sc, const std::vector<reco::TrackRef> tr, Charge q, const LorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ), const Point & convVtx = Point( 0, 0, 0 ) ):  
    RecoCandidate( q, p4, vtx, 22 * q ), superCluster_(sc), tracks_(tr), theConversionVertex_(convVtx) {

  makePairInvariantMass();
  makePairCotThetaSeparation();
  makePairMomentum();
  makePairMomentumEta();
  makePairMomentumPhi();
  makePairPtOverEtSC();
  makeEoverP() ;

 }


ConvertedPhoton::~ConvertedPhoton() { }

ConvertedPhoton * ConvertedPhoton::clone() const { 
  return new ConvertedPhoton( * this ); 
}

reco::SuperClusterRef ConvertedPhoton::superCluster() const {
  return superCluster_;
}




std::vector<reco::TrackRef>  ConvertedPhoton::tracks() const { 
   return tracks_;
}


TrackRef ConvertedPhoton::track( size_t ind ) const {
  return tracks_[ind];
}



bool ConvertedPhoton::isConverted() const {
  
  if ( this->nTracks() > 0) 
    return true;
  else
    return false;
}


void  ConvertedPhoton::makePairInvariantMass() {
  invMass_=-99.;
  const float mElec= 0.000511;
  if ( nTracks()==2 ) {
    double px= tracks()[0]->innerMomentum().x() + tracks()[1]->innerMomentum().x();
    double py= tracks()[0]->innerMomentum().y() + tracks()[1]->innerMomentum().y();
    double pz= tracks()[0]->innerMomentum().z() + tracks()[1]->innerMomentum().z();  
    double mom1=tracks()[0]->innerMomentum().Mag2() ;
    double mom2=tracks()[1]->innerMomentum().Mag2() ;
    double e = sqrt( mom1+ mElec*mElec ) + sqrt( mom2 + mElec*mElec );
    invMass_= ( e*e - px*px -py*py - pz*pz);
  }
  

 

}

void  ConvertedPhoton::makePairCotThetaSeparation()  {
  dCotTheta_=-99.;
  
  if ( nTracks()==2 ) {
    double theta1=tracks()[0]->innerMomentum().Theta();
    double theta2=tracks()[1]->innerMomentum().Theta();
    dCotTheta_ =  1./tan(theta1) - 1./tan(theta2) ;
  }
}


void  ConvertedPhoton::makePairMomentum()  {

  double px=0.;
  double py=0.;
  double pz=0.;
  
  if ( nTracks()==2 ) {
    px= tracks()[0]->innerMomentum().x() + tracks()[1]->innerMomentum().x();
    py= tracks()[0]->innerMomentum().y() + tracks()[1]->innerMomentum().y();
    pz= tracks()[0]->innerMomentum().z() + tracks()[1]->innerMomentum().z();  

  } else if (  nTracks()==1 ) {
    px= tracks()[0]->innerMomentum().x() ;
    py= tracks()[0]->innerMomentum().y() ;
    pz= tracks()[0]->innerMomentum().z() ;
  }


  momTracks_= GlobalVector(px,py,pz);

}


void  ConvertedPhoton::makePairMomentumEta()  {

  etaTracks_=-99.;
  
  if ( nTracks() > 0 ) {
    etaTracks_= pairMomentum().eta();
  } 

}

void  ConvertedPhoton::makePairMomentumPhi()  {

  phiTracks_=-99.;
  
  if ( nTracks() > 0 ) {
    phiTracks_= pairMomentum().phi();
  } 

}




void  ConvertedPhoton::makePairPtOverEtSC()  {

  ptOverEtSC_=-99.;
  double etaSC=superCluster()->eta();
  double EtSC= superCluster()->energy()/cosh(etaSC);
  
   if ( nTracks() ==2 ) {
    ptOverEtSC_= (pairMomentum().perp()) /EtSC; 
   }

}




void  ConvertedPhoton::makeEoverP()  {

  ep_=-99.;
  if ( nTracks() > 0  ) 
    ep_=  this->superCluster()->energy()/this->pairMomentum().mag();
  
}


bool ConvertedPhoton::overlap( const Candidate & c ) const {
  const RecoCandidate * o = dynamic_cast<const RecoCandidate *>( & c );
  return ( o != 0 && 
	   ( checkOverlap( superCluster(), o->superCluster() ) )
	   );
  return false;
}
   
