#include "FWCore/Framework/interface/MakerMacros.h"

#include "CommonTools/UtilAlgos/interface/ObjectSelectorStream.h"
#include "CommonTools/UtilAlgos/interface/SortCollectionSelector.h"
#include "RecoHI/HiTracking/interface/BestVertexComparator.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

namespace reco
{
	namespace modules
	{
		
		// define your producer name
		typedef ObjectSelectorStream<
			SortCollectionSelector<
			reco::VertexCollection,
			GreaterByTracksSize<reco::Vertex>
			> 
			> HIBestVertexSelection;
		
		// declare the module as plugin
		DEFINE_FWK_MODULE( HIBestVertexSelection );
	}
}
