/* \class MuonSelector
 *
 * Selects muon with a configurable string-based cut.
 * Saves clones of the selected muons 
 * Warning: this module can read anything that inherits from reco::Muon, but it will
 *   only clone the reco::Muon part of the object, the rest is lost.
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
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

 typedef SingleObjectSelector<
           edm::View<reco::Muon>, 
           StringCutObjectSelector<reco::Muon>,
           reco::MuonCollection
         > MuonSelector;

DEFINE_FWK_MODULE( MuonSelector );
