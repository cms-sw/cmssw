#include "FWCore/Framework/interface/MakerMacros.h"

#include "CommonTools/RecoAlgos/interface/TrackFullCloneSelectorBase.h"
#include "CommonTools/RecoAlgos/interface/TrackWithVertexSelector.h"

namespace reco { 
  namespace modules {

    typedef TrackFullCloneSelectorBase< ::TrackWithVertexSelector > TrackWithVertexSelector;

    DEFINE_FWK_MODULE(TrackWithVertexSelector);

} }
