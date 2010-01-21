#include "PhysicsTools/UtilAlgos/interface/CollectionAdder.h"
#include "DataFormats/Common/interface/ValueMap.h"

typedef CollectionAdder<edm::ValueMap<int> > ValeMapIntMerger;
typedef CollectionAdder<edm::ValueMap<unsigned int> > ValeMapUIntMerger;
typedef CollectionAdder<edm::ValueMap<float> > ValeMapFloatMerger;
typedef CollectionAdder<edm::ValueMap<double> > ValeMapDoubleMerger;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( ValeMapIntMerger );
DEFINE_FWK_MODULE( ValeMapUIntMerger );
DEFINE_FWK_MODULE( ValeMapFloatMerger );
DEFINE_FWK_MODULE( ValeMapDoubleMerger );

