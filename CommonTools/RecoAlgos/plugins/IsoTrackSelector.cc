#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "DataFormats/PatCandidates/interface/IsolatedTrack.h"


 typedef SingleObjectSelector<
           pat::IsolatedTrackCollection, 
           StringCutObjectSelector<pat::IsolatedTrack> 
         > IsoTrackSelector;

DEFINE_FWK_MODULE( IsoTrackSelector );
