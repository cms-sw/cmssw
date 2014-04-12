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
                                          double intRadiusBarrel,
                                          double intRadiusEndcap,
                                          double stripBarrel,
                                          double stripEndcap,
                                          double ptLow,
                                          double lip,
                                          double drb,
                                          const reco::TrackCollection* trackCollection,
                                          reco::TrackBase::Point beamPoint,
                                          const std::string &dzOptionString) :
  extRadius_(extRadius),
  intRadiusBarrel_(intRadiusBarrel),
  intRadiusEndcap_(intRadiusEndcap),
  stripBarrel_(stripBarrel),
  stripEndcap_(stripEndcap),
  ptLow_(ptLow),
  lip_(lip),
  drb_(drb),
  trackCollection_(trackCollection),
  beamPoint_(beamPoint)
{
    setDzOption(dzOptionString);
}

ElectronTkIsolation::~ElectronTkIsolation ()
{}

std::pair<int,double> ElectronTkIsolation::getIso(const reco::GsfElectron* electron) const {
  return getIso(&(*(electron->gsfTrack())));
}

// unified acces to isolations
std::pair<int,double> ElectronTkIsolation::getIso(const reco::Track* tmpTrack) const  
{
  int counter  =0 ;
  double ptSum =0.;
  //Take the electron track
  //reco::GsfTrackRef tmpTrack = electron->gsfTrack() ;
  math::XYZVector tmpElectronMomentumAtVtx = (*tmpTrack).momentum () ; 
  double tmpElectronEtaAtVertex = (*tmpTrack).eta();


  for ( reco::TrackCollection::const_iterator itrTr  = (*trackCollection_).begin() ; 
	itrTr != (*trackCollection_).end()   ; 
	++itrTr ) {

    double this_pt  = (*itrTr).pt();
    if ( this_pt < ptLow_ ) continue;


    double dzCut = 0;
    switch( dzOption_ ) {
        case egammaisolation::EgammaTrackSelector::dz : dzCut = fabs( (*itrTr).dz() - (*tmpTrack).dz() ); break;
        case egammaisolation::EgammaTrackSelector::vz : dzCut = fabs( (*itrTr).vz() - (*tmpTrack).vz() ); break;
        case egammaisolation::EgammaTrackSelector::bs : dzCut = fabs( (*itrTr).dz(beamPoint_) - (*tmpTrack).dz(beamPoint_) ); break;
        case egammaisolation::EgammaTrackSelector::vtx: dzCut = fabs( (*itrTr).dz(tmpTrack->vertex()) ); break;
        default : dzCut = fabs( (*itrTr).vz() - (*tmpTrack).vz() ); break;
    }
    if (dzCut > lip_ ) continue;
    if (fabs( (*itrTr).dxy(beamPoint_) ) > drb_   ) continue;
    double dr = ROOT::Math::VectorUtil::DeltaR(itrTr->momentum(),tmpElectronMomentumAtVtx) ;
    double deta = (*itrTr).eta() - tmpElectronEtaAtVertex;
    if (fabs(tmpElectronEtaAtVertex) < 1.479) { 
    	if ( fabs(dr) < extRadius_ && fabs(dr) >= intRadiusBarrel_ && fabs(deta) >= stripBarrel_)
      	{
	    ++counter ;
	    ptSum += this_pt;
      	}
    }
    else {
        if ( fabs(dr) < extRadius_ && fabs(dr) >= intRadiusEndcap_ && fabs(deta) >= stripEndcap_)
        {
            ++counter ;
            ptSum += this_pt;
        }
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

