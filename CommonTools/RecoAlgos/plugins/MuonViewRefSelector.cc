/* \class MuonRefToBaseSelector
 *
 * Selects muon with a configurable string-based cut.
 * Saves references to the selected muons
 *
 * \author: Luca Lista, INFN
 *
 * usage:
 *
 * module bestMuons = MuonRefToBaseSelector {
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
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/RefToBaseVector.h"

typedef SingleObjectSelector<
       edm::View<reco::Muon>, 
       StringCutObjectSelector<reco::Muon>,
       edm::RefToBaseVector<reco::Muon>
     > MuonViewRefSelector;

DEFINE_FWK_MODULE( MuonViewRefSelector );
