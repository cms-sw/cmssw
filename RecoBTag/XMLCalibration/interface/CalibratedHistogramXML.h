#ifndef CalibratedHistogramXML_H
#define CalibratedHistogramXML_H
#include <xercesc/dom/DOM.hpp>
#include "RecoBTag/XMLCalibration/interface/CalibratedObject.h"
#include "CondFormats/BTauObjects/interface/CalibratedHistogram.h"
#include <vector>
#include <xercesc/dom/DOMNode.hpp>

/**
* This class implements some methods of the CalibratedObject.
* This class does not provide methdos for calibration, i.e.
*  [start|update|finish]Calibration() functions.
* If you want to use it in a calibration program you have to 
* implement those methods in a child class.
*/

class CalibratedHistogramXML : public CalibratedHistogram, CalibratedObject {
public:
  typedef XERCES_CPP_NAMESPACE::DOMElement DOMElement;
  typedef XERCES_CPP_NAMESPACE::DOMNode DOMNode;

  CalibratedHistogramXML() {}
  CalibratedHistogramXML(const CalibratedHistogram &h) : CalibratedHistogram(h) {}
  CalibratedHistogramXML(const std::vector<float> &ulimits) : CalibratedHistogram(ulimits) {}
  ~CalibratedHistogramXML() override {}

  void read(DOMElement *dom) override;

  void write(DOMElement *dom) const override;

  std::string name() const override { return "CalibratedHistogramXML"; }
};

#endif
