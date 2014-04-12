#include "FWCore/Framework/interface/MakerMacros.h"

/*
  This is an example of using the primary vertex selector PVSelector from the PhysiocsTools/SelectorUtils 
  to wrap it into an EDFilter. The resulting module is an event EDFilter, filtering events based on the 
  PV selection.
*/
#include "PhysicsTools/SelectorUtils/interface/PVSelector.h"
#include "PhysicsTools/UtilAlgos/interface/EDFilterWrapper.h"
typedef edm::FilterWrapper<PVSelector> PrimaryVertexFilter;
DEFINE_FWK_MODULE(PrimaryVertexFilter);

/*
  This is an example of using the primary vertex object selector PVObjectSelector from the PhysicsTools/
  SelectorUtils to wrap it into an EDProducer. The resulting module is an EDProducer filtering objects 
  based on the PV selection. A new collection will be produced containing only the selectr objects.
*/
#include "PhysicsTools/SelectorUtils/interface/PVObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/EDFilterObjectWrapper.h"
typedef edm::FilterObjectWrapper<PVObjectSelector, std::vector<reco::Vertex> > PrimaryVertexObjectFilter;
DEFINE_FWK_MODULE(PrimaryVertexObjectFilter);
