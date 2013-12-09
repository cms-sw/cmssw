/** \class reco::modules::TrackingParticleSelector
 *
 *  Filter to select TrackingParticles according to pt, rapidity, tip, lip, number of hits, pdgId
 *
 *  \author Ian Tomalin, RAL
 *
 *  $Date: 2010/02/25 00:28:56 $
 *  $Revision: 1.2 $
 */

#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "SimGeneral/TrackingAnalysis/interface/TrackingParticleSelector.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"

namespace reco {
  namespace modules {
    typedef SingleObjectSelector<TrackingParticleCollection,::TrackingParticleSelector,TrackingParticleRefVector> 
    TrackingParticleRefSelector ;

    DEFINE_FWK_MODULE( TrackingParticleRefSelector );
  }
}
