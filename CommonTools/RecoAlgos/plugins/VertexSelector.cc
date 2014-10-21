/* \class VertexSelector
 *
 * Selects vertices with a configurable string-based cut.
 * Saves clones of the selected vertices 
 *
 * \author: Luca Lista, INFN
 *
 * usage:
 *
 * module bestVertices = VertexSelector {
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

 using VertexSelector = SingleObjectSelectorStream<
           reco::VertexCollection, 
           StringCutObjectSelector<reco::Vertex> 
         >;

DEFINE_FWK_MODULE(VertexSelector);
