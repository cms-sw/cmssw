// -*- C++ -*-
//
// Package:     Tracks
// Class  :     FWPhase2TrackerCluster1DDetProxyBuilder
//
//

#include "TEveGeoNode.h"

#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"

#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"
#include "DataFormats/DetId/interface/DetId.h"

class FWPhase2TrackerCluster1DDetProxyBuilder : public FWProxyBuilderBase
{
public:
  FWPhase2TrackerCluster1DDetProxyBuilder() {}
  ~FWPhase2TrackerCluster1DDetProxyBuilder() override {}
  
  REGISTER_PROXYBUILDER_METHODS();

private:
  using FWProxyBuilderBase::build;
  void build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*) override;
  FWPhase2TrackerCluster1DDetProxyBuilder(const FWPhase2TrackerCluster1DDetProxyBuilder&) = delete;
  const FWPhase2TrackerCluster1DDetProxyBuilder& operator=(const FWPhase2TrackerCluster1DDetProxyBuilder&) = delete;
};

void FWPhase2TrackerCluster1DDetProxyBuilder::build( const FWEventItem* iItem, TEveElementList* product , const FWViewContext*)
{
  const Phase2TrackerCluster1DCollectionNew* pixels = nullptr;
  
  iItem->get(pixels);
  
  if( ! pixels ) 
    return;
  
  const FWGeometry* geom = iItem->getGeom();
  
  for( Phase2TrackerCluster1DCollectionNew::const_iterator set = pixels->begin(), setEnd = pixels->end();
       set != setEnd; ++set) {
    unsigned int id = set->detId();
    DetId detid(id);
    
    if( geom->contains( detid )) {
      const edmNew::DetSet<Phase2TrackerCluster1D> & clusters = *set;
	
      for( edmNew::DetSet<Phase2TrackerCluster1D>::const_iterator itc = clusters.begin(), edc = clusters.end(); 
           itc != edc; ++itc ) {
        TEveGeoShape* shape = geom->getEveShape(detid);
	
        if ( shape ) {
          shape->SetMainTransparency(50);
          setupAddElement(shape, product);
        }
      }
    }
  }
}

REGISTER_FWPROXYBUILDER( FWPhase2TrackerCluster1DDetProxyBuilder, Phase2TrackerCluster1DCollectionNew, "Phase2TrackerCluster1DDets", FWViewType::kAll3DBits | FWViewType::kAllRPZBits );
