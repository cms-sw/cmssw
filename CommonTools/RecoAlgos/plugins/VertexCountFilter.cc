/* \class TrackCountFilter
 *
 * Filters events if at least N vertices
 *
 * \author: Steven Lowette
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "CommonTools/UtilAlgos/interface/ObjectCountFilter.h"

 typedef ObjectCountFilter<
           reco::VertexCollection
         >::type VertexCountFilter;

DEFINE_FWK_MODULE( VertexCountFilter );
