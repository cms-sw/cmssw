
#include "L1Trigger/L1THGCal/interface/HGCalTriggerTools.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"

void
HGCalTriggerTools::
eventSetup(const edm::EventSetup& es)
{
  edm::ESHandle<HGCalTriggerGeometryBase> geometry;
  es.get<CaloGeometryRecord>().get("", geometry);

  eeLayers_ = geometry->eeTopology().dddConstants().layers(true);
  fhLayers_ = geometry->fhTopology().dddConstants().layers(true);
  bhLayers_ = geometry->bhTopology().dddConstants()->getMaxDepth(1);
  totalLayers_ =  eeLayers_ + fhLayers_ + bhLayers_;
}

unsigned
HGCalTriggerTools::
layers(ForwardSubdetector type) const
{
  unsigned layers = 0;
  switch(type)
  {
    case ForwardSubdetector::HGCEE:
      layers = eeLayers_;
      break;
    case ForwardSubdetector::HGCHEF:
      layers = fhLayers_;
      break;
    case ForwardSubdetector::HGCHEB:
      layers = bhLayers_;
      break;
    case ForwardSubdetector::ForwardEmpty:
      layers = totalLayers_;
      break;
    default:
      break;
  };
  return layers;
}

unsigned
HGCalTriggerTools::
layerWithOffset(unsigned id) const
{
  HGCalDetId detid(id);
  unsigned layer = 0;
  switch(detid.subdetId())
  {
    case ForwardSubdetector::HGCEE:
      layer = detid.layer();
      break;
    case ForwardSubdetector::HGCHEF:
      layer = eeLayers_ + detid.layer();
      break;
    case ForwardSubdetector::HGCHEB:
      layer = eeLayers_ + fhLayers_ + detid.layer();
      break;
    default:
      break;
  };
  return layer;
}
