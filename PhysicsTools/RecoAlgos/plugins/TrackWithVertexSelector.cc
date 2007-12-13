#include "FWCore/Framework/interface/MakerMacros.h"

#include "PhysicsTools/RecoAlgos/interface/TrackFullCloneSelectorBase.h"
#include "PhysicsTools/RecoAlgos/interface/TrackWithVertexSelector.h"

namespace reco { 
  namespace modules {

    typedef TrackFullCloneSelectorBase< ::TrackWithVertexSelector > TrackWithVertexSelector;

    DEFINE_FWK_MODULE(TrackWithVertexSelector);

} }
