#include "RecoBTag/XMLCalibration/interface/CalibratedHistogramXML.h"
#include "RecoBTag/XMLCalibration/interface/CalibrationXML.h"
#include <iostream>

using namespace std;
void CalibratedHistogramXML::read(XERCES_CPP_NAMESPACE::DOMElement *dom) {
  binValues.clear();
  binULimits.clear();
  int size = CalibrationXML::readAttribute<int>(dom, "size");

  DOMNode *n1 = dom->getFirstChild();
  int bin;
  for (bin = 0; bin < size; bin++) {
    while ((n1->getNodeType() != DOMNode::ELEMENT_NODE) && (n1 != nullptr))
      n1 = n1->getNextSibling();
    if (n1) {
      DOMElement *binElement = (DOMElement *)n1;
      binValues.push_back(CalibrationXML::readAttribute<double>(binElement, "value"));

      binULimits.push_back(CalibrationXML::readAttribute<double>(binElement, "uLimit"));
      n1 = n1->getNextSibling();
    }
  }
  if (bin > 0)
    binValues.push_back(CalibrationXML::readAttribute<int>(dom, "overFlowValue"));

  limits = Range(binULimits.front(), binULimits.back());
  totalValid = false;
}

void CalibratedHistogramXML::write(XERCES_CPP_NAMESPACE::DOMElement *dom) const {
  int size = binULimits.size();
  CalibrationXML::writeAttribute(dom, "size", size);
  DOMElement *binElement;
  for (int bin = 0; bin < size; bin++) {
    binElement = CalibrationXML::addChild(dom, "Bin");
    CalibrationXML::writeAttribute(binElement, "value", binValues[bin]);
    CalibrationXML::writeAttribute(binElement, "uLimit", binULimits[bin]);
  }
  CalibrationXML::writeAttribute(dom, "overFlowValue", binValues[size]);
}
