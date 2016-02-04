/* \class SuperClusterRefSelector
 *
 * Selects super-cluster with a configurable string-based cut.
 * Saves references to the selected super-clusters 
 *
 * \author: Luca Lista, INFN
 *
 * usage:
 *
 * module bestSuperClusters = SuperClusterRefSelector {
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
           StringCutObjectSelector<reco::SuperCluster>,
           reco::SuperClusterRefVector
         > SuperClusterRefSelector;

DEFINE_FWK_MODULE( SuperClusterRefSelector );
