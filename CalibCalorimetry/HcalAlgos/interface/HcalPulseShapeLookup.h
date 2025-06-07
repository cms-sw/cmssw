#ifndef CalibCalorimetry_HcalAlgos_HcalPulseShapeLookup_h
#define CalibCalorimetry_HcalAlgos_HcalPulseShapeLookup_h

#include <vector>
#include <string>
#include <utility>

#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseShape.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"

// This class must be sufficiently similar in its interface to HcalPulseShapes
// so that both of them can be used as parameters of the same templated code.
// However, this one is designed for use with a more efficient pulse shape
// lookup scheme and can accommodate more pulse shapes.
class HcalPulseShapeLookup {
public:
  typedef HcalPulseShape Shape;
  typedef std::pair<std::string, Shape> LabeledShape;

  HcalPulseShapeLookup(const std::vector<LabeledShape>& shapes,
                       const std::vector<int>& channelToTypeLookup,
                       const HcalTopology* htopo);

  inline unsigned nShapeTypes() const { return theShapes_.size(); }
  const Shape& getShape(int shapeType) const;
  const std::string& getLabel(int shapeType) const;

  int getShapeType(unsigned linearizedChannelNumber) const;
  const Shape& getChannelShape(unsigned linearizedChannelNumber) const;
  const std::string& getChannelLabel(unsigned linearizedChannelNumber) const;

  int getShapeType(const DetId& id) const;
  const Shape& getChannelShape(const DetId& id) const;
  const std::string& getChannelLabel(const DetId& id) const;

private:
  std::vector<LabeledShape> theShapes_;
  std::vector<int> shapeTypes_;
  // We do not own the pointer
  const HcalTopology* htopo_;
};

#endif  // CalibCalorimetry_HcalAlgos_HcalPulseShapeLookup_h
