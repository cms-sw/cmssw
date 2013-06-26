/* \class PhotonRefSelector
 *
 * Selects photon with a configurable string-based cut.
 * Saves references to the selected photons 
 *
 * \author: Luca Lista, INFN
 *
 * usage:
 *
 * module bestPhotons = PhotonRefSelector {
 *   src = photons
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
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"

 typedef SingleObjectSelector<
           reco::PhotonCollection, 
           StringCutObjectSelector<reco::Photon>,
           reco::PhotonRefVector
         > PhotonRefSelector;

DEFINE_FWK_MODULE( PhotonRefSelector );
