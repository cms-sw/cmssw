/* \class SuperClusterSelector
 *
 * Selects super-cluster with a configurable string-based cut.
 * Saves clones of the selected super-clusters 
 *
 * \author: Luca Lista, INFN
 *
 * usage:
 *
 * module bestSuperClusters = SuperClusterSelector {
 *   src = hybridSuperClusters
 *   string cut = "energy > 20 & abs( eta ) < 2"
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
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

 typedef SingleObjectSelector<
           reco::SuperClusterCollection, 
           StringCutObjectSelector<reco::SuperCluster>
         > SuperClusterSelector;

DEFINE_FWK_MODULE( SuperClusterSelector );
