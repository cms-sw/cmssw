/** \class reco::modules::TrackingParticleSelector
 *
 *  Filter to select TrackingParticles according to pt, rapidity, tip, lip, number of hits, pdgId
 *
 *  \author Giuseppe Cerati, INFN
 *
 *  $Date: 2010/02/25 00:28:57 $
 *  $Revision: 1.2 $
 */

#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "SimGeneral/TrackingAnalysis/interface/TrackingParticleSelector.h"

namespace reco {
  namespace modules {
    typedef SingleObjectSelector<TrackingParticleCollection,::TrackingParticleSelector> 
    TrackingParticleSelector ;

    DEFINE_FWK_MODULE( TrackingParticleSelector );
  }
}
