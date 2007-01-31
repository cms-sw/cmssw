/* \class ElectronCountFilter
 *
 * Filters events if at least N electrons
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "PhysicsTools/UtilAlgos/interface/ObjectCountFilter.h"

 typedef ObjectCountFilter<
           reco::ElectronCollection
         > ElectronCountFilter;

DEFINE_FWK_MODULE( ElectronCountFilter );
