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
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/Candidate/interface/Particle.h"

using namespace ROOT::Math::VectorUtil ;

PhotonTkIsolation::PhotonTkIsolation (double extRadius,
				      double intRadius,
				      double etLow,
				      double lip,
				      const reco::TrackCollection* trackCollection)   :
  extRadius_(extRadius),
  intRadius_(intRadius),
  etLow_(etLow),
  lip_(lip),
  trackCollection_(trackCollection)  
{
}

PhotonTkIsolation::~PhotonTkIsolation ()
{
}



// unified acces to isolations
std::pair<int,double> PhotonTkIsolation::getIso(const reco::Candidate* photon ) const  
{
  int counter  =0 ;
  double ptSum =0.;


  //Take the SC position
  reco::SuperClusterRef sc = photon->get<reco::SuperClusterRef>();
  math::XYZPoint theCaloPosition = sc.get()->position();
  reco::Particle::Point vtxPos = photon->vertex();
  math::XYZVector mom (theCaloPosition.x() - vtxPos.x(),
		    theCaloPosition.y() - vtxPos.y(),
		    theCaloPosition.z() - vtxPos.z());

  //loop over tracks
  for(reco::TrackCollection::const_iterator trItr = trackCollection_->begin(); trItr != trackCollection_->end(); ++trItr){

    //check z-distance of vertex 
    if (fabs( (*trItr).dz() - photon->vertex().z() ) >= lip_ ) continue ;

    math::XYZVector tmpTrackMomentumAtVtx = (*trItr).momentum () ;
    double this_pt  = (*trItr).pt();
    if ( this_pt < etLow_ ) 
      continue ;  
    double dr = DeltaR(tmpTrackMomentumAtVtx,mom) ;
    if(fabs(dr) < extRadius_ &&
       fabs(dr) >= intRadius_ )
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

