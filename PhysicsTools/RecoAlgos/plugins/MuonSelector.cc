/* \class MuonSelector
 *
 * Selects muon with a configurable string-based cut.
 * Saves clones of the selected muons 
 *
 * \author: Luca Lista, INFN
 *
 * usage:
 *
 * module bestMuons = MuonSelector {
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
#include "PhysicsTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

 typedef SingleObjectSelector<
           reco::MuonCollection, 
           StringCutObjectSelector<reco::Muon> 
         > MuonSelector;

DEFINE_FWK_MODULE( MuonSelector );
