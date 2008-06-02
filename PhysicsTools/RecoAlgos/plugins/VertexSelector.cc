/* \class VertexSelector
 *
 * Selects track with a configurable string-based cut.
 * Saves clones of the selected tracks 
 *
 * \author: Luca Lista, INFN
 *
 * usage:
 *
 * module bestTracks = VertexSelector {
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
#include "PhysicsTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

 typedef SingleObjectSelector<
           reco::VertexCollection, 
           StringCutObjectSelector<reco::Vertex> 
         > VertexSelector;

DEFINE_FWK_MODULE( VertexSelector );
