#include "L1Trigger/L1TMuon/src/Phase2/GeometryHelpers.h"
#include "L1Trigger/CSCCommonTrigger/interface/CSCConstants.h"

GlobalPoint
L1TMuon::GeometryHelpers::globalPositionOfME0LCT(const ME0Geometry* geometry,
                                                 const ME0Segment& seg)
{
  const ME0DetId& Id(seg.me0DetId());
  const LocalPoint& localPos(seg.localPosition());
  const ME0Chamber* chamber(geometry->chamber(Id));
  const GlobalPoint& pos(chamber->toGlobal(localPos));
  return pos;
}

GlobalPoint
L1TMuon::GeometryHelpers::globalPositionOfGEMPad(const GEMGeometry* geometry,
                                                 const GEMPadDigi& gempad,
                                                 const GEMDetId& id)
{
  const GEMEtaPartition* gemRoll(geometry->etaPartition(id));
  const LocalPoint& lpGEM(gemRoll->centreOfPad(gempad.pad()));
  const GlobalPoint& pos(gemRoll->toGlobal(lpGEM));
  return pos;
}

GlobalPoint
L1TMuon::GeometryHelpers::globalPositionOfGEMCoPad(const GEMGeometry* geometry,
                                                   const GEMCoPadDigi& gempad,
                                                   const GEMDetId& id)
{
  const GEMEtaPartition* gemRoll(geometry->etaPartition(id));
  const LocalPoint& lpGEM(gemRoll->centreOfPad( (float) ((gempad.pad(1) + gempad.pad(2))/2.) ));
  const GlobalPoint& pos(gemRoll->toGlobal(lpGEM));
  return pos;
}

GlobalPoint
L1TMuon::GeometryHelpers::globalPositionOfCSCLCT(const CSCGeometry* geometry,
                                                 const CSCCorrelatedLCTDigi& stub,
                                                 const CSCDetId& cscId)
{
  const CSCDetId key_id(cscId.endcap(), cscId.station(), cscId.ring(),
                        cscId.chamber(), CSCConstants::KEY_CLCT_LAYER);
  const auto& cscChamber = geometry->chamber(cscId);
  const float fractional_strip = stub.getFractionalStrip();
  const auto& layer_geo = cscChamber->layer(CSCConstants::KEY_CLCT_LAYER)->geometry();
  // LCT::getKeyWG() also starts from 0
  const float wire = layer_geo->middleWireOfGroup(stub.getKeyWG() + 1);
  // local point as the intersection between strip and wire
  const LocalPoint& csc_intersect = layer_geo->intersectionOfStripAndWire(fractional_strip, wire);
  const GlobalPoint& csc_gp = cscChamber->toGlobal(csc_intersect);

  return csc_gp;
}
