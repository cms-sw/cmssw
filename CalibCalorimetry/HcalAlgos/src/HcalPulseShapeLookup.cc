#include <cassert>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>

#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseShapeLookup.h"

HcalPulseShapeLookup::HcalPulseShapeLookup(const std::vector<LabeledShape>& shapes,
                                           const std::vector<int>& channelToTypeLookup,
                                           const HcalTopology* htopo)
    : theShapes_(shapes), shapeTypes_(channelToTypeLookup), htopo_(htopo) {
  assert(htopo_);
}

const HcalPulseShapeLookup::Shape& HcalPulseShapeLookup::getShape(const int shapeType) const {
  return std::get<2>(theShapes_.at(shapeType));
}

const std::string& HcalPulseShapeLookup::getLabel(const int shapeType) const {
  return std::get<0>(theShapes_.at(shapeType));
}

float HcalPulseShapeLookup::getTimeShift(const int shapeType) const { return std::get<1>(theShapes_.at(shapeType)); }

int HcalPulseShapeLookup::getShapeType(const unsigned linearizedChannelNumber) const {
  return shapeTypes_.at(linearizedChannelNumber);
}

int HcalPulseShapeLookup::getShapeType(const DetId& id) const { return getShapeType(htopo_->detId2denseId(id)); }

const HcalPulseShapeLookup::Shape& HcalPulseShapeLookup::getChannelShape(const unsigned linearizedChannelNumber) const {
  return std::get<2>(theShapes_.at(shapeTypes_.at(linearizedChannelNumber)));
}

const std::string& HcalPulseShapeLookup::getChannelLabel(const unsigned linearizedChannelNumber) const {
  return std::get<0>(theShapes_.at(shapeTypes_.at(linearizedChannelNumber)));
}

float HcalPulseShapeLookup::getChannelTimeShift(const unsigned linearizedChannelNumber) const {
  return std::get<1>(theShapes_.at(shapeTypes_.at(linearizedChannelNumber)));
}

const HcalPulseShapeLookup::Shape& HcalPulseShapeLookup::getChannelShape(const DetId& id) const {
  return getChannelShape(htopo_->detId2denseId(id));
}

const std::string& HcalPulseShapeLookup::getChannelLabel(const DetId& id) const {
  return getChannelLabel(htopo_->detId2denseId(id));
}

float HcalPulseShapeLookup::getChannelTimeShift(const DetId& id) const {
  return getChannelTimeShift(htopo_->detId2denseId(id));
}

void HcalPulseShapeLookup::dumpToTxt(const std::string& filename, const unsigned precision) const {
  std::ofstream of(filename.c_str());
  if (!of.is_open()) {
    throw cms::Exception("HcalPulseShapeLookup::dumpToTxt: file opening error")
        << "Failed to open output file \"" << filename << '"' << std::endl;
  }
  if (precision)
    of.precision(precision);

  const int nShapes = theShapes_.size();
  const unsigned nShapeTupes = shapeTypes_.size();
  const unsigned nChannelsMapped = nShapeTupes - std::count(shapeTypes_.begin(), shapeTypes_.end(), -1);
  of << "# nShapes = " << nShapes << " nChannelsMapped = " << nChannelsMapped << '\n';
  for (int i = 0; i < nShapes; ++i) {
    const std::vector<float>& data = getShape(i).data();
    const unsigned len = data.size();
    of << i << ' ' << getLabel(i) << ' ' << getTimeShift(i) << ' ' << len;
    for (const float d : data)
      of << ' ' << d;
    of << '\n';
  }

  of << "####\n";
  std::map<HcalSubdetector, std::string> subDetNames;
  subDetNames[HcalBarrel] = "HB";
  subDetNames[HcalEndcap] = "HE";
  for (unsigned densId = 0; densId < nShapeTupes; ++densId)
    if (shapeTypes_[densId] >= 0) {
      const HcalDetId hcalId(htopo_->denseId2detId(densId));
      const HcalSubdetector subdet = hcalId.subdet();
      assert(subDetNames.find(subdet) != subDetNames.end());
      of << subDetNames[subdet] << std::setw(4) << hcalId.ieta() << std::setw(4) << hcalId.iphi() << std::setw(3)
         << hcalId.depth() << ' ' << shapeTypes_[densId] << '\n';
    }

  if (!of.good()) {
    throw cms::Exception("HcalPulseShapeLookup::dumpToTxt: failed to write the pulse shapes");
  }
}
