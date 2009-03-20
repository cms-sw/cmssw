
#include "DataFormats/EgammaCandidates/interface/GsfElectronCore.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"

#include <cmath>

using namespace reco ;

GsfElectronCore::GsfElectronCore
 ()
 : isEcalDriven_(false), isTrackerDriven_(false)
 {}

GsfElectronCore::GsfElectronCore
 ( const GsfTrackRef & gsfTrack )
 : gsfTrack_(gsfTrack), isEcalDriven_(false), isTrackerDriven_(false)
 {
  edm::RefToBase<TrajectorySeed> seed = gsfTrack_->extra()->seedRef() ;
  if (seed.isNull())
   { edm::LogError("GsfElectronCore")<<"The GsfTrack has no seed ?!" ; }
  else
   {
    ElectronSeedRef elseed = seed.castTo<ElectronSeedRef>() ;
    if (elseed.isNull())
     { edm::LogError("GsfElectronCore")<<"The GsfTrack seed is not an ElectronSeed ?!" ; }
    else
     {
      if (!(elseed->caloCluster().isNull()))
       { isEcalDriven_ = true ; }
      if (!(elseed->ctfTrack().isNull()))
       { isTrackerDriven_ = true ; }
     }
   }
 }
