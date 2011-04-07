#include "PhysicsTools/UtilAlgos/interface/EDFilterWrapper.h"
#include "PhysicsTools/UtilAlgos/interface/EDFilterObjectWrapper.h"
#include "PhysicsTools/SelectorUtils/interface/PVSelector.h"
#include "PhysicsTools/SelectorUtils/interface/PVObjectSelector.h"

typedef edm::FilterWrapper<PVSelector> PrimaryVertexFilter;
typedef edm::FilterObjectWrapper<PVObjectSelector, std::vector<reco::Vertex> > PrimaryVertexObjectFilter;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PrimaryVertexFilter);
DEFINE_FWK_MODULE(PrimaryVertexObjectFilter);
