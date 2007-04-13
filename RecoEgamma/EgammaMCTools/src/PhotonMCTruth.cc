
#include "RecoEgamma/EgammaMCTools/interface/PhotonMCTruth.h"

#include <iostream>


PhotonMCTruth::PhotonMCTruth(int isAConversion,HepLorentzVector v, float rconv, float zconv,
					       HepLorentzVector convVertex,  
                                               HepLorentzVector pV,  std::vector<const SimTrack *> tracks  ) :
  isAConversion_(isAConversion),
  thePhoton_(v), theR_(rconv), theZ_(zconv), theConvVertex_(convVertex), 
  thePrimaryVertex_(pV), tracks_(tracks)  {

  std::cout << " PhotonMCTruth constructor " << std::endl; 
  //  ecalImpactPositions(); 
  //basicClusters();
  
}




float PhotonMCTruth::invMass() const {
  float invm=-999,px,py,pz,e;
  const float mElec = 0.000511;

  /*
  if(theTracks_.size() > 1) {
    
    
    px= theTracks_[0]->momentumAtVertex().x() +  theTracks_[1]->momentumAtVertex().x();
    py= theTracks_[0]->momentumAtVertex().y() +  theTracks_[1]->momentumAtVertex().y();
    pz= theTracks_[0]->momentumAtVertex().z() +  theTracks_[1]->momentumAtVertex().z();
    e  = sqrt( (theTracks_[0]->momentumAtVertex().mag())*(theTracks_[0]->momentumAtVertex().mag()) + mElec*mElec) + 
      sqrt( (theTracks_[1]->momentumAtVertex().mag())*(theTracks_[1]->momentumAtVertex().mag()) + mElec*mElec);
    invm=(e*e - px*px -py*py - pz*pz);
    if ( invm > 0.0 ) { 
      invm=sqrt(invm);
    } else { 
      invm=-sqrt(-invm); 
    }
    
    
  }
 
  */ 
    return invm;
}

float PhotonMCTruth::ptTrack1() const{
  /*
  if(theTracks_.size() > 0) {
    return theTracks_[0]->momentum().perp()*theTracks_[0]->charge();
  } else {
    return 0;
  }

  */
  return 0;

}

float PhotonMCTruth::ptTrack2() const{

  /*
  if(theTracks_.size() > 1) {
    return theTracks_[1]->momentum().perp()*theTracks_[1]->charge();
  } else {
    return 0;
  }

  */
  return 0;

}

/*
vector<GlobalPoint> PhotonMCTruth::getHitPositions(const TkSimTrack* const &theTkSimTrack)  {
  vector<GlobalPoint> theHitPositions;
  TkSimTrack::SimHitContainer theHits = theTkSimTrack->hits();
  TkSimTrack::SimHitContainer::const_iterator i;
  for(i = theHits.begin(); i != theHits.end(); ++i) {
    theHitPositions.push_back((**i).globalPosition());
  }
  if (theHitPositions.size() != theHits.size()) cout << "ERROR in no. of hits!!!" << endl;
  return theHitPositions;
}

*/



/*
void PhotonMCTruth::basicClusters() {
  //cout << "theTracks_.size() = " << theTracks_.size() << endl;
  if(theTracks_.size() > 0) {
    //cout << "Conversion Vertex: " << (theTracks_[0])->stateAtVertex().globalPosition() << endl;
    //cout << endl << "First track:" << endl;
    //cout << " Momentum = " << theTracks_[0]->momentum().perp();
    //cout << " Charge = " << theTracks_[0]->charge();
    //printTrackHits(theTracks_[0]);
    theBasicClusters_.push_back(bcFromTrack(theTracks_[0]));
  }
  if(theTracks_.size() > 1) {
    //cout << endl << "Second track:" << endl;
    //cout << " Momentum = " << theTracks_[1]->momentum().perp();
    //cout << " Charge = " << theTracks_[1]->charge();
    //printTrackHits(theTracks_[1]);
    theBasicClusters_.push_back(bcFromTrack(theTracks_[1]));
  }
}

*/

/*
const EgammaBasicCluster * PhotonMCTruth::bcFromTrack(const TkSimTrack * theTkSimTrack) const {

  //Find the position of the outermost SimHit on the SimTrack
  TkSimTrack::SimHitContainer theHits = theTkSimTrack->hits();
  GlobalPoint theLastHitPosition;
  if (theHits.size() > 0) {
    theLastHitPosition = theHits[theHits.size()-1]->globalPosition();
  } else {
    theLastHitPosition = theTkSimTrack->vertex()->position();
  }

  const EgammaBasicCluster * bcTrack = 0;
  float distToTrack=999.;
  //Loop over all basic clusters in the event

  vector<string> dum;
  EgammaItr<EgammaBasicCluster> bcItr(dum,0);
  GlobalPoint closestApproach;
  while (bcItr.next()){
    GlobalPoint bcPos(bcItr->position().x(),bcItr->position().y(),bcItr->position().z());
    //cout << "  bc at " << bcPos << endl;
    // Only consider basic clusters within a reasonable window around the 
    // eta and phi of the outermost hit
    if(fabs(bcPos.eta()-theLastHitPosition.eta()) < .2 && 
       (fabs(bcPos.phi()-theLastHitPosition.phi()) < .5 || 
	fabs(bcPos.phi()-theLastHitPosition.phi())-2.*M_PI < .5)) {
      //cout << "    bc eta, phi: " << bcPos.eta() << " " << bcPos.phi() 
      //     << ", last hit eta, phi: " << theLastHitPosition.eta() 
      //     << " " << theLastHitPosition.phi() << endl;
      if( theTkSimTrack->stateAtPoint(bcPos).isValid()) {
	// Find the closest approach of the extrapolated track to the cluster 
        // position
	closestApproach = theTkSimTrack->stateAtPoint(bcPos).globalPosition();
	//cout << "    distToTrack: " << (bcPos-closestApproach).mag() << endl;
	//cout << "    clostestApproachPosn: " << closestApproach 
        //     << ", eta = " << closestApproach.eta() 
        //     << ", phi = " << closestApproach.phi() << endl;
	//Find the closest cluster to the track
	if((bcPos-closestApproach).mag() < distToTrack) {
	  bcTrack = &(*bcItr);
	  distToTrack = (bcPos-closestApproach).mag();
	}
      }
    }
  }
  if(bcTrack != 0) {
    GlobalPoint bcPos(bcTrack->position().x(),bcTrack->position().y(),bcTrack->position().z());
    //cout << "Closest bc: " << bcPos 
    //     << "  Track hits ECAL at " << closestApproach 
    //     << "  distToTrack = " << distToTrack << endl << endl;
    if (distToTrack > 10.) bcTrack = 0;
  }
  return bcTrack;
}

*/


/*
void PhotonMCTruth::printTrackHits (const TkSimTrack * theTkSimTrack) const {
  TkSimTrack::SimHitContainer theHits = theTkSimTrack->hits();
  TkSimTrack::SimHitContainer::const_iterator i;
  int i1=0;
  cout << "  No. of hits = " << theHits.size() << endl;
  for(i = theHits.begin(); i != theHits.end(); ++i) {
    ++i1;
    cout << "  hit " << i1 << " at " << (**i).globalPosition() << endl;
  }
}

*/


/*

void PhotonMCTruth::ecalImpactPositions() {
  if(theTracks_.size() > 0) {
    TrajectoryStateOnSurface stateAtECAL = theTracks_[0]->stateOnSurface(ECALSurfaces::barrel());
    if (!stateAtECAL.isValid() || abs(stateAtECAL.globalPosition().eta())>1.479) {
      if (theConvVertex_.pseudoRapidity() > 0.) {
	stateAtECAL = 
	  theTracks_[0]->stateOnSurface(ECALSurfaces::positiveEtaEndcap());
      } else {
	stateAtECAL = 
	  theTracks_[0]->stateOnSurface(ECALSurfaces::negativeEtaEndcap());
      }
    }
    ecalPosTrack1_ = stateAtECAL.isValid() ? stateAtECAL.globalPosition() : GlobalPoint(0.,0.,0.);
    //cout << "1st track impacts ECAL at " << ecalPosTrack1_ << endl;
  }
  if(theTracks_.size() > 1) {
    TrajectoryStateOnSurface stateAtECAL = theTracks_[1]->stateOnSurface(ECALSurfaces::barrel());
    if (!stateAtECAL.isValid() || abs(stateAtECAL.globalPosition().eta())>1.479) {
      if (theConvVertex_.pseudoRapidity() > 0.) {
	stateAtECAL = 
	  theTracks_[1]->stateOnSurface(ECALSurfaces::positiveEtaEndcap());
      } else {
	stateAtECAL = 
	  theTracks_[1]->stateOnSurface(ECALSurfaces::negativeEtaEndcap());
      }
    }
    ecalPosTrack2_ = stateAtECAL.isValid() ? stateAtECAL.globalPosition() : GlobalPoint(0.,0.,0.);
    //cout << "2nd track impacts ECAL at " << ecalPosTrack2_ << endl;
  }
}
*/
