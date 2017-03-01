#include "FWCore/Framework/interface/MakerMacros.h"

#include "CommonTools/UtilAlgos/interface/ObjectSelectorStream.h"
#include "RecoHI/HiTracking/plugins/HIProtoTrackSelector.h"

namespace reco
{
	namespace modules
	{
		
		// define your producer name
		typedef ObjectSelectorStream<HIProtoTrackSelector> HIProtoTrackSelection;
		
		// declare the module as plugin
		DEFINE_FWK_MODULE( HIProtoTrackSelection );
	}
}
