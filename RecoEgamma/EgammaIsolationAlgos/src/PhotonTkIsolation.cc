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

using namespace ROOT::Math::VectorUtil;


PhotonTkIsolation::PhotonTkIsolation (double extRadius,
                                      double intRadiusBarrel,
                                      double intRadiusEndcap,
                                      double stripBarrel,
                                      double stripEndcap,
                                      double etLow,
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
  etLow_(etLow),
  lip_(lip),
  drb_(drb),
  trackCollection_(trackCollection),
  beamPoint_(beamPoint)
{
    setDzOption(dzOptionString);
}

PhotonTkIsolation::~PhotonTkIsolation ()
{
}



// unified acces to isolations
std::pair<int,double> PhotonTkIsolation::getIso(const reco::Candidate* photon ) const  
{
  int counter  =0 ;
  double ptSum =0.;


  //Take the photon position
  math::XYZVector mom= photon->momentum();
  double photonEta = photon->eta();

  //loop over tracks
  for(reco::TrackCollection::const_iterator trItr = trackCollection_->begin(); trItr != trackCollection_->end(); ++trItr){

    //check z-distance of vertex 
    double dzCut = 0;
    switch( dzOption_ ) {
        case egammaisolation::EgammaTrackSelector::dz : dzCut = fabs( (*trItr).dz() - photon->vertex().z() ); break;
        case egammaisolation::EgammaTrackSelector::vz : dzCut = fabs( (*trItr).vz() - photon->vertex().z() ); break;
        case egammaisolation::EgammaTrackSelector::bs : dzCut = fabs( (*trItr).dz(beamPoint_) - photon->vertex().z() ); break;
        case egammaisolation::EgammaTrackSelector::vtx: dzCut = fabs( (*trItr).dz(photon->vertex())); break;
        default : dzCut = fabs( (*trItr).vz() - photon->vertex().z() ); break;
    }
    if (dzCut > lip_ ) continue;

    math::XYZVector tmpTrackMomentumAtVtx = (*trItr).momentum () ;
    double this_pt  = (*trItr).pt();
    if ( this_pt < etLow_ ) continue ;  
    if (fabs( (*trItr).dxy(beamPoint_) ) > drb_   ) continue;// only consider tracks from the main vertex 
    double dr = ROOT::Math::VectorUtil::DeltaR(tmpTrackMomentumAtVtx,mom) ;
    double deta = (*trItr).eta() - photonEta ;
    if (fabs(photonEta) < 1.479) {
    	if(fabs(dr) < extRadius_ && fabs(dr) >= intRadiusBarrel_ && fabs(deta) >= stripBarrel_) 
      	{
	    ++counter;
	    ptSum += this_pt;
      	}
    }
    else {
        if(fabs(dr) < extRadius_ && fabs(dr) >= intRadiusEndcap_ && fabs(deta) >= stripEndcap_)
        {
            ++counter;
            ptSum += this_pt;
        }
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

