/* \class PixelMatchGsfElectronRefSelector
 *
 * Selects PixelMatchGsfElectron with a configurable string-based cut.
 * Saves clones of the selected PixelMatchGsfElectrons 
 *
 * \author: Luca Lista, INFN
 *
 * usage:
 *
 * module bestPixelMatchGsfElectrons = PixelMatchGsfElectronRefSelector {
 *   src = pixelMatchGsfElectron
 *   string cut = "pt > 20 & abs( eta ) < 2"
 * }
 *
 * for more details about the cut syntax, see the documentation
 * page below:
 *
 *   https://twiki.cern.ch/twiki/bin/view/CMS/SWGuidePhysicsCutParser
 *
 *
 */

#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectronFwd.h"

 typedef SingleObjectSelector<
           reco::PixelMatchGsfElectronCollection, 
           StringCutObjectSelector<reco::PixelMatchGsfElectron>,
           reco::PixelMatchGsfElectronRefVector
         > PixelMatchGsfElectronRefSelector;

DEFINE_FWK_MODULE( PixelMatchGsfElectronRefSelector );
