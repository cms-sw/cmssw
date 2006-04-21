// $Id: Electron.cc,v 1.1 2006/04/09 15:39:25 rahatlou Exp $
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
using namespace reco;

Electron::Electron( const Vector & calo, const Vector & track, short charge,
		    double eHadOverEcal, short isolation, short pixelLines ) :
  caloMomentum_( calo ), trackMomentum_( track ), charge_( charge ),
  eHadOverEcal_( eHadOverEcal ), isolation_( isolation ), pixelLines_( pixelLines ) {
}

/*
Electron::Electron( const SuperClusterRef & calo, TrackRef track,
		    double eHadOverEcal, short isolation, short pixelLines ) :
  caloMomentum_( calo->momentum() ), trackMomentum_( track->momentum() ),
  eHadOverEcal_( eHadOverEcal ), isolation_( isolation ), pixelLines_( pixelLines ),
  superCluster_( calo ), track_( track ) {
}
*/
