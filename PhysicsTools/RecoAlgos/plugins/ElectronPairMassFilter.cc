/* \class ElectronPairMassFilter
 *
 * Filters events if at least N electrons pairs
 * with an invariant mass within a specified range
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "PhysicsTools/UtilAlgos/interface/MasslessInvariantMass.h"
#include "PhysicsTools/UtilAlgos/interface/RangeObjectPairSelector.h"
#include "PhysicsTools/UtilAlgos/interface/ObjectPairFilter.h"

typedef ObjectPairFilter<
          reco::ElectronCollection,
          RangeObjectPairSelector<
            reco::Electron, 
            MasslessInvariantMass<reco::Electron> >
        > ElectronPairMassFilter;

DEFINE_FWK_MODULE( ElectronPairMassFilter );
