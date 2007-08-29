#include "RecoTracker/NuclearSeedGenerator/interface/NuclearSeedsEDProducer.h"
//define this as a plug-in
DEFINE_FWK_MODULE(NuclearSeedsEDProducer);

#include "RecoTracker/NuclearSeedGenerator/interface/NuclearSeedsToTrackAssociationEDProducer.h"
DEFINE_ANOTHER_FWK_MODULE(NuclearSeedsToTrackAssociationEDProducer);