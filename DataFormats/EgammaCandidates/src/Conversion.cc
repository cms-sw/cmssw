#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/TrackReco/interface/Track.h" 
#include "DataFormats/CaloRecHit/interface/CaloCluster.h" 
#include "CLHEP/Units/GlobalPhysicalConstants.h"

using namespace reco;


Conversion::Conversion(  const reco::CaloClusterPtrVector sc, 
			 const std::vector<reco::TrackRef> tr, 
			 const std::vector<math::XYZPointF> trackPositionAtEcal, 
			 const reco::Vertex  & convVtx,
			 const std::vector<reco::CaloClusterPtr> & matchingBC,
                         const float DCA,
			 const std::vector<math::XYZPointF> & innPoint,
			 const std::vector<math::XYZVectorF> & trackPin,
			 const std::vector<math::XYZVectorF> & trackPout,
                         const float mva,
			 ConversionAlgorithm algo):  
			 

  caloCluster_(sc), tracks_(tr), 
  thePositionAtEcal_(trackPositionAtEcal), 
  theConversionVertex_(convVtx), 
  theMatchingBCs_(matchingBC), 
  theMinDistOfApproach_(DCA),
  theTrackInnerPosition_(innPoint),
  theTrackPin_(trackPin),
  theTrackPout_(trackPout),
  nSharedHits_(0),  
  theMVAout_(mva),
  algorithm_(algo),
  qualityMask_(0)
 { 
   
 }




Conversion::Conversion(  const reco::CaloClusterPtrVector sc, 
			 const std::vector<edm::RefToBase<reco::Track> > tr, 
			 const std::vector<math::XYZPointF> trackPositionAtEcal, 
			 const reco::Vertex  & convVtx,
			 const std::vector<reco::CaloClusterPtr> & matchingBC,
                         const float DCA,
			 const std::vector<math::XYZPointF> & innPoint,
			 const std::vector<math::XYZVectorF> & trackPin,
			 const std::vector<math::XYZVectorF> & trackPout,
                         const std::vector<uint8_t> nHitsBeforeVtx,                  
                         const std::vector<Measurement1DFloat> & dlClosestHitToVtx,
                         uint8_t nSharedHits,
                         const float mva,
			 ConversionAlgorithm algo):  
			 

  caloCluster_(sc), trackToBaseRefs_(tr), 
  thePositionAtEcal_(trackPositionAtEcal), 
  theConversionVertex_(convVtx), 
  theMatchingBCs_(matchingBC), 
  theMinDistOfApproach_(DCA),
  theTrackInnerPosition_(innPoint),
  theTrackPin_(trackPin),
  theTrackPout_(trackPout),
  nHitsBeforeVtx_(nHitsBeforeVtx),
  dlClosestHitToVtx_(dlClosestHitToVtx),
  nSharedHits_(nSharedHits),
  theMVAout_(mva),
  algorithm_(algo),
  qualityMask_(0)
 { 
   
 }




Conversion::Conversion(  const reco::CaloClusterPtrVector sc, 
			 const std::vector<reco::TrackRef> tr, 
			 const reco::Vertex  & convVtx,
			 ConversionAlgorithm algo):  
  caloCluster_(sc), tracks_(tr), 
  theConversionVertex_(convVtx),
  nSharedHits_(0),
  algorithm_(algo),
  qualityMask_(0)
 { 


  theMinDistOfApproach_ = 9999.;
  theMVAout_ = 9999.;
  thePositionAtEcal_.push_back(math::XYZPointF(0.,0.,0.));
  thePositionAtEcal_.push_back(math::XYZPointF(0.,0.,0.));
  theTrackInnerPosition_.push_back(math::XYZPointF(0.,0.,0.));
  theTrackInnerPosition_.push_back(math::XYZPointF(0.,0.,0.));
  theTrackPin_.push_back(math::XYZVectorF(0.,0.,0.));
  theTrackPin_.push_back(math::XYZVectorF(0.,0.,0.));
  theTrackPout_.push_back(math::XYZVectorF(0.,0.,0.));
  theTrackPout_.push_back(math::XYZVectorF(0.,0.,0.));

   
 }


Conversion::Conversion(  const reco::CaloClusterPtrVector sc, 
			 const std::vector<edm::RefToBase<reco::Track> >  tr, 
			 const reco::Vertex  & convVtx,
			 ConversionAlgorithm algo):  
  caloCluster_(sc), trackToBaseRefs_(tr), 
  theConversionVertex_(convVtx), 
  nSharedHits_(0),
  algorithm_(algo),
  qualityMask_(0)
 { 


  theMinDistOfApproach_ = 9999.;
  theMVAout_ = 9999.;
  thePositionAtEcal_.push_back(math::XYZPointF(0.,0.,0.));
  thePositionAtEcal_.push_back(math::XYZPointF(0.,0.,0.));
  theTrackInnerPosition_.push_back(math::XYZPointF(0.,0.,0.));
  theTrackInnerPosition_.push_back(math::XYZPointF(0.,0.,0.));
  theTrackPin_.push_back(math::XYZVectorF(0.,0.,0.));
  theTrackPin_.push_back(math::XYZVectorF(0.,0.,0.));
  theTrackPout_.push_back(math::XYZVectorF(0.,0.,0.));
  theTrackPout_.push_back(math::XYZVectorF(0.,0.,0.));

   
 }



Conversion::Conversion() { 

  algorithm_=0;
  qualityMask_=0;
  theMinDistOfApproach_ = 9999.;
  nSharedHits_ = 0;
  theMVAout_ = 9999.;
  thePositionAtEcal_.push_back(math::XYZPointF(0.,0.,0.));
  thePositionAtEcal_.push_back(math::XYZPointF(0.,0.,0.));
  theTrackInnerPosition_.push_back(math::XYZPointF(0.,0.,0.));
  theTrackInnerPosition_.push_back(math::XYZPointF(0.,0.,0.));
  theTrackPin_.push_back(math::XYZVectorF(0.,0.,0.));
  theTrackPin_.push_back(math::XYZVectorF(0.,0.,0.));
  theTrackPout_.push_back(math::XYZVectorF(0.,0.,0.));
  theTrackPout_.push_back(math::XYZVectorF(0.,0.,0.));
    
}


Conversion::~Conversion() { }


std::string const Conversion::algoNames[] = { "undefined","ecalSeeded","trackerOnly","mixed","pflow"};  

Conversion::ConversionAlgorithm Conversion::algoByName(const std::string &name){
  ConversionAlgorithm size = algoSize;
  int index = std::find(algoNames, algoNames+size, name)-algoNames;
  if(index == size) return undefined;

  return ConversionAlgorithm(index);
}

Conversion * Conversion::clone() const { 
  return new Conversion( * this ); 
}


std::vector<edm::RefToBase<reco::Track> >  Conversion::tracks() const { 
  if (trackToBaseRefs_.size() ==0 ) {
 
    for (std::vector<reco::TrackRef>::const_iterator ref=tracks_.begin(); ref!=tracks_.end(); ref++ ) 
      {
	edm::RefToBase<reco::Track> tt(*ref);
	trackToBaseRefs_.push_back(tt);
	
      }  
  }

  return trackToBaseRefs_;
}



bool Conversion::isConverted() const {
  
  if ( this->nTracks() == 2 ) 
    return true;
  else
    return false;
}

double Conversion::pairInvariantMass() const{
  double invMass=-99.;
  const float mElec= 0.000511;
  if ( nTracks()==2 ) {
    double px= tracksPin()[0].x() +  tracksPin()[1].x();
    double py= tracksPin()[0].y() +  tracksPin()[1].y();
    double pz= tracksPin()[0].z() +  tracksPin()[1].z();
    double mom1= tracksPin()[0].Mag2();
    double mom2= tracksPin()[1].Mag2();
    double e = sqrt( mom1+ mElec*mElec ) + sqrt( mom2 + mElec*mElec );
    invMass= ( e*e - px*px -py*py - pz*pz);
    if ( invMass>0) invMass = sqrt(invMass);
    else 
      invMass = -1;
  }
  
  return invMass;
}

double  Conversion::pairCotThetaSeparation() const  {
  double dCotTheta=-99.;
  
  if ( nTracks()==2 ) {
    double theta1=this->tracksPin()[0].Theta();
    double theta2=this->tracksPin()[1].Theta();
    dCotTheta =  1./tan(theta1) - 1./tan(theta2) ;
  }

  return dCotTheta;

}


math::XYZVectorF  Conversion::pairMomentum() const  {
  
  if ( nTracks()==2 ) {
    return this->tracksPin()[0] +  this->tracksPin()[1];
  }
  return math::XYZVectorF(0.,0.,0.);



}


math::XYZTLorentzVectorF Conversion::refittedPair4Momentum() const  {

  math::XYZTLorentzVectorF p4;
  if ( this->conversionVertex().isValid() ) 
    p4 = this->conversionVertex().p4( 0.000511, 0.5);

  return p4;


}



math::XYZVectorF  Conversion::refittedPairMomentum() const  {

  if (  this->conversionVertex().isValid() ) {
    return this->refittedPair4Momentum().Vect();
  }
  return math::XYZVectorF(0.,0.,0.);

}



double  Conversion::EoverP() const  {


  double ep=-99.;

  if ( nTracks() > 0  ) {
    unsigned int size= this->caloCluster().size();
    float etot=0.;
    for ( unsigned int i=0; i<size; i++) {
      etot+= caloCluster()[i]->energy();
    }
    if (this->pairMomentum().Mag2() !=0) ep= etot/sqrt(this->pairMomentum().Mag2());
  }



  return ep;  

}



double  Conversion::EoverPrefittedTracks() const  {


  double ep=-99.;
 
  if ( nTracks() > 0  ) {
    unsigned int size= this->caloCluster().size();
    float etot=0.;
    for ( unsigned int i=0; i<size; i++) {
      etot+= caloCluster()[i]->energy();
    }
    if (this->refittedPairMomentum().Mag2() !=0) ep= etot/sqrt(this->refittedPairMomentum().Mag2());
  }



  return ep;  

}

 

std::vector<double>  Conversion::tracksSigned_d0() const  {
  std::vector<double> result;

  for (unsigned int i=0; i< nTracks(); i++)   
    result.push_back(tracks()[i]->d0()* tracks()[i]->charge()) ;

  return result;


}

double  Conversion::dPhiTracksAtVtx() const  {
  double result=-99;
  if  ( nTracks()==2 ) {
    result = tracksPin()[0].phi() - tracksPin()[1].phi();
    if( result   > pi)  { result = result - twopi;}
    if( result  < -pi)  { result = result + twopi;}
  }

  return result;


}

double  Conversion::dPhiTracksAtEcal() const  {
  double result =-99.;
  
  if (  nTracks()==2  && bcMatchingWithTracks()[0].isNonnull() && bcMatchingWithTracks()[1].isNonnull() ) {
    
    float recoPhi1 = ecalImpactPosition()[0].phi();
    if( recoPhi1   > pi)  { recoPhi1 = recoPhi1 - twopi;}
    if( recoPhi1  < -pi)  { recoPhi1 = recoPhi1 + twopi;}

    float recoPhi2 = ecalImpactPosition()[1].phi();
    if( recoPhi2   > pi)  { recoPhi2 = recoPhi2 - twopi;}
    if( recoPhi2  < -pi)  { recoPhi2 = recoPhi2 + twopi;}

    result = recoPhi1 -recoPhi2;

    if( result   > pi)  { result = result - twopi;}
    if( result  < -pi)  { result = result + twopi;}

  }

  return result;


}

double  Conversion::dEtaTracksAtEcal() const  {
  double result=-99.;


  if ( nTracks()==2 && bcMatchingWithTracks()[0].isNonnull() && bcMatchingWithTracks()[1].isNonnull() ) {

    result =ecalImpactPosition()[0].eta() - ecalImpactPosition()[1].eta();

  }



  return result;


}

double Conversion::dxy(const math::XYZPoint& myBeamSpot) const {

  const reco::Vertex &vtx = conversionVertex();
  if (!vtx.isValid()) return -9999.;

  math::XYZVectorF mom = refittedPairMomentum();
  
  double dxy = (-(vtx.x() - myBeamSpot.x())*mom.y() + (vtx.y() - myBeamSpot.y())*mom.x())/mom.rho();
  return dxy;  
  
}

double Conversion::dz(const math::XYZPoint& myBeamSpot) const {

  const reco::Vertex &vtx = conversionVertex();
  if (!vtx.isValid()) return -9999.;

  math::XYZVectorF mom = refittedPairMomentum();
  
  double dz = (vtx.z()-myBeamSpot.z()) - ((vtx.x()-myBeamSpot.x())*mom.x()+(vtx.y()-myBeamSpot.y())*mom.y())/mom.rho() * mom.z()/mom.rho();
  return dz;  
  
}

double Conversion::lxy(const math::XYZPoint& myBeamSpot) const {

  const reco::Vertex &vtx = conversionVertex();
  if (!vtx.isValid()) return -9999.;

  math::XYZVectorF mom = refittedPairMomentum();
  
  double dbsx = vtx.x() - myBeamSpot.x();
  double dbsy = vtx.y() - myBeamSpot.y();
  double lxy = (mom.x()*dbsx + mom.y()*dbsy)/mom.rho();
  return lxy;  
  
}

double Conversion::lz(const math::XYZPoint& myBeamSpot) const {

  const reco::Vertex &vtx = conversionVertex();
  if (!vtx.isValid()) return -9999.;

  math::XYZVectorF mom = refittedPairMomentum();
  
  double lz = (vtx.z() - myBeamSpot.z())*mom.z()/std::abs(mom.z());
  return lz;  
  
}

