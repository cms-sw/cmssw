//*****************************************************************************
// File:      ElectronTkIsolation.cc
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
#include "RecoEgamma/EgammaIsolationAlgos/interface/ElectronTkIsolation.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"

using namespace ROOT::Math::VectorUtil ;


ElectronTkIsolation::ElectronTkIsolation (double extRadius,
			  double intRadius,
			  double ptLow,
			  double lip,
			  const reco::TrackCollection* trackCollection)   :
  extRadius_(extRadius),
  intRadius_(intRadius),
  ptLow_(ptLow),
  lip_(lip),
  trackCollection_(trackCollection)  
{
}

ElectronTkIsolation::~ElectronTkIsolation ()
{
}

// unified acces to isolations
std::pair<int,double> ElectronTkIsolation::getIso(const reco::GsfElectron* electron) const  
{
  int counter  =0 ;
  double ptSum =0.;
  //Take the electron track
  reco::GsfTrackRef tmpTrack = electron->gsfTrack() ;
  math::XYZVector tmpElectronMomentumAtVtx = (*tmpTrack).momentum () ; 

  for ( reco::TrackCollection::const_iterator itrTr  = (*trackCollection_).begin() ; 
                                              itrTr != (*trackCollection_).end()   ; 
	   			              ++itrTr ) 
    {
	math::XYZVector tmpTrackMomentumAtVtx = (*itrTr).momentum () ; 
	double this_pt  = (*itrTr).pt();
	if ( this_pt < ptLow_ ) 
	  continue ;  
	if (fabs( (*itrTr).dz() - (*tmpTrack).dz() ) > lip_ )
          continue ;
	double dr = DeltaR(tmpTrackMomentumAtVtx,tmpElectronMomentumAtVtx) ;
	if ( fabs(dr) < extRadius_ && 
	     fabs(dr) >= intRadius_ )
	  {
	    ++counter ;
	    ptSum += this_pt;
	  }
    }//end loop over tracks                 

  std::pair<int,double> retval;
  retval.first  = counter;
  retval.second = ptSum;

  return retval;
}


int ElectronTkIsolation::getNumberTracks (const reco::GsfElectron* electron) const
{  
  //counter for the tracks in the isolation cone
  return getIso(electron).first ;
}

double ElectronTkIsolation::getPtTracks (const reco::GsfElectron* electron) const
{
  return getIso(electron).second ;
}

