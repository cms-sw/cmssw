/* \class MuonRefSelector
 *
 * Selects muon with a configurable string-based cut.
 * Saves references to the selected muons
 *
 * \author: Luca Lista, INFN
 *
 * usage:
 *
 * module bestMuons = MuonRefSelector {
 *   src = muons
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
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

 typedef SingleObjectSelector<
           reco::MuonCollection, 
           StringCutObjectSelector<reco::Muon>,
           reco::MuonRefVector
         > MuonRefSelector;

DEFINE_FWK_MODULE( MuonRefSelector );
