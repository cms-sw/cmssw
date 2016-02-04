/* \class TrackSelector
 *
 * Selects track with a configurable string-based cut.
 * Saves clones of the selected tracks, and also
 * clones track extras and rec-hits
 *
 * \author: Luca Lista, INFN
 *
 * usage:
 *
 * module bestTracks = TrackFullCloneSelector {
 *   src = ctfWithMaterialTracks
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
#include "CommonTools/RecoAlgos/interface/TrackSelector.h"
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

 typedef SingleObjectSelector<
           reco::TrackCollection, 
           StringCutObjectSelector<reco::Track> 
         > TrackFullCloneSelector;

DEFINE_FWK_MODULE(TrackFullCloneSelector);

