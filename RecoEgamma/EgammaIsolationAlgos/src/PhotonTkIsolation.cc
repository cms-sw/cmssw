//*****************************************************************************
// File:      PhotonTkIsolation.cc
// ----------------------------------------------------------------------------
// OrigAuth:  Matthias Mozer
// Institute: IIHE-VUB
//=============================================================================
//*****************************************************************************
//C++ includes
#include <vector>
#include <functional>

//ROOT includes
#include <Math/VectorUtil.h>

//CMSSW includes
#include "RecoEgamma/EgammaIsolationAlgos/interface/PhotonTkIsolation.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"

using namespace ROOT::Math::VectorUtil ;

PhotonTkIsolation::PhotonTkIsolation (double extRadius,
				      double intRadius,
				      double etLow,
				      const reco::TrackCollection* trackCollection)   :
  extRadius_(extRadius),
  intRadius_(intRadius),
  etLow_(etLow),
  trackCollection_(trackCollection)  
{
}

PhotonTkIsolation::~PhotonTkIsolation ()
{
}



// unified acces to isolations
std::pair<int,double> PhotonTkIsolation::getIso(const reco::Candidate* photon) const  
{
  int counter  =0 ;
  double ptSum =0.;


  //Take the SC position
  reco::SuperClusterRef sc = photon->get<reco::SuperClusterRef>();
  math::XYZPoint theCaloPosition = sc.get()->position();
  math::XYZVector mom (theCaloPosition.x () ,
		    theCaloPosition.y () ,
		    theCaloPosition.z () );

  //loop over tracks
  for(reco::TrackCollection::const_iterator trItr = trackCollection_->begin(); trItr != trackCollection_->end(); ++trItr){
    math::XYZVector tmpTrackMomentumAtVtx = (*trItr).innerMomentum () ; 
    double this_pt  = sqrt( tmpTrackMomentumAtVtx.Perp2 () );
    if ( this_pt < etLow_ ) 
      continue ;  
    double dr = DeltaR(tmpTrackMomentumAtVtx,mom) ;
    if(fabs(dr) < extRadius_ &&
       fabs(dr) > intRadius_ )
      {
	++counter;
	ptSum += this_pt;
      }
    
  }//end loop over tracks

  std::pair<int,double> retval;
  retval.first  = counter;
  retval.second = ptSum;

  return retval;
}

int PhotonTkIsolation::getNumberTracks (const reco::Candidate* photon) const
{  
  //counter for the tracks in the isolation cone
  return getIso(photon).first ;
}

double PhotonTkIsolation::getPtTracks (const reco::Candidate* photon) const
{
  return getIso(photon).second ;
}

