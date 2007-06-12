/**
 *  See header file for a description of this class.
 *
 *  $Date: 2007/06/08 12:01:31 $
 *  $Revision: 1.1 $
 *  \author D. Pagano - University of Pavia & INFN Pavia
 */


#include "RecoMuon/MuonSeedGenerator/src/RPCSeedFinder.h"
#include "RecoMuon/MuonSeedGenerator/src/RPCSeedHits.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"

#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "DataFormats/GeometryVector/interface/Pi.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/GeometryVector/interface/CoordinateSets.h"
#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"

#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include <iomanip>

using namespace std;

typedef MuonTransientTrackingRecHit::MuonRecHitPointer MuonRecHitPointer;
typedef MuonTransientTrackingRecHit::ConstMuonRecHitPointer ConstMuonRecHitPointer;
typedef MuonTransientTrackingRecHit::MuonRecHitContainer MuonRecHitContainer;

RPCSeedFinder::RPCSeedFinder(){}

vector<TrajectorySeed> RPCSeedFinder::seeds(const edm::EventSetup& eSetup) const {

  cout << "[RPCSeedFinder] --> seeds class called" << endl;

  vector<TrajectorySeed> theSeeds;

  RPCSeedHits barrel;

  int num_bar = 0;
  for ( MuonRecHitContainer::const_iterator iter = theRhits.begin(); iter!= theRhits.end(); iter++ ){
      barrel.add(*iter);
      num_bar++;
  }

  if ( num_bar ) {
    cout << "[RPCSeedFinder] --> Barrel Seeds " << num_bar << endl;
    theSeeds.push_back(barrel.seed(eSetup));
  }
  
  return theSeeds;
}

