/* \class GsfElectronRefSelector
 *
 * Selects GsfElectron with a configurable string-based cut.
 * Saves clones of the selected GsfElectrons 
 *
 * \author: Luca Lista, INFN
 *
 * usage:
 *
 * module bestGsfElectrons = GsfElectronRefSelector {
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
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

 typedef SingleObjectSelector<
           reco::GsfElectronCollection, 
           StringCutObjectSelector<reco::GsfElectron>,
           reco::GsfElectronRefVector
         > GsfElectronRefSelector;

DEFINE_FWK_MODULE( GsfElectronRefSelector );
