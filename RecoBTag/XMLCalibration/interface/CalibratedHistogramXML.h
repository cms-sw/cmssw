#ifndef CalibratedHistogramXML_H
#define CalibratedHistogramXML_H
#include <xercesc/dom/DOM.hpp>
#include "RecoBTag/XMLCalibration/interface/CalibratedObject.h"
#include "CondFormats/BTagObjects/interface/CalibratedHistogram.h"
#include <vector>
#include <xercesc/dom/DOMNode.hpp>


/**
* This class implements some methods of the CalibratedObject.
* This class does not provide methdos for calibration, i.e.
*  [start|update|finish]Calibration() functions.
* If you want to use it in a calibration program you have to 
* implement those methods in a child class.
*/

class CalibratedHistogramXML:public CalibratedHistogram, CalibratedObject
{
public:
  CalibratedHistogramXML() {} 
  CalibratedHistogramXML(const CalibratedHistogram &h):CalibratedHistogram(h) {} 
  virtual ~CalibratedHistogramXML() {} 

   
  void read (XERCES_CPP_NAMESPACE::DOMElement * dom);
  
  void write (XERCES_CPP_NAMESPACE::DOMElement * dom) const;

  std::string name () const
  {
    return "CalibratedHistogramXML";
  }
};

#endif
