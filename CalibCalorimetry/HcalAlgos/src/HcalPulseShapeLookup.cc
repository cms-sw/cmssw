#include <cassert>

#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseShapeLookup.h"

HcalPulseShapeLookup::HcalPulseShapeLookup(const std::vector<LabeledShape>& shapes,
                                           const std::vector<int>& channelToTypeLookup,
                                           const HcalTopology* htopo)
    : theShapes_(shapes), shapeTypes_(channelToTypeLookup), htopo_(htopo) {
  assert(htopo_);
}

const HcalPulseShapeLookup::Shape& HcalPulseShapeLookup::getShape(const int shapeType) const {
  return theShapes_.at(shapeType).second;
}

const std::string& HcalPulseShapeLookup::getLabel(const int shapeType) const { return theShapes_.at(shapeType).first; }

int HcalPulseShapeLookup::getShapeType(const unsigned linearizedChannelNumber) const {
  return shapeTypes_.at(linearizedChannelNumber);
}

int HcalPulseShapeLookup::getShapeType(const DetId& id) const { return getShapeType(htopo_->detId2denseId(id)); }

const HcalPulseShapeLookup::Shape& HcalPulseShapeLookup::getChannelShape(const unsigned linearizedChannelNumber) const {
  return theShapes_.at(shapeTypes_.at(linearizedChannelNumber)).second;
}

const std::string& HcalPulseShapeLookup::getChannelLabel(const unsigned linearizedChannelNumber) const {
  return theShapes_.at(shapeTypes_.at(linearizedChannelNumber)).first;
}

const HcalPulseShapeLookup::Shape& HcalPulseShapeLookup::getChannelShape(const DetId& id) const {
  return getChannelShape(htopo_->detId2denseId(id));
}

const std::string& HcalPulseShapeLookup::getChannelLabel(const DetId& id) const {
  return getChannelLabel(htopo_->detId2denseId(id));
}
