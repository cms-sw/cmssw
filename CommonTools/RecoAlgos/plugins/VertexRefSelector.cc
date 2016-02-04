/* \class VertexRefSelector
 *
 * Selects vertices with a configurable string-based cut.
 * Saves references to the selected vertices 
 *
 * \author: Luca Lista, INFN
 *
 * usage:
 *
 * module bestVertices = VertexRefSelector {
 *   src = ctfWithMaterialTracks
 *   string cut = "chiSquared < 5"
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
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

 typedef SingleObjectSelector<
           reco::VertexCollection, 
           StringCutObjectSelector<reco::Vertex>,
           reco::VertexRefVector
         > VertexRefSelector;

DEFINE_FWK_MODULE(VertexRefSelector);
