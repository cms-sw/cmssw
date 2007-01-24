#include "RecoBTag/XMLCalibration/interface/CalibratedHistogramXML.h"
#include "RecoBTag/XMLCalibration/interface/CalibrationXML.h"
#include <iostream>

using namespace std;
void  CalibratedHistogramXML::read (XERCES_CPP_NAMESPACE::DOMElement * dom)
  {
    m_normalization=0; 
    m_binValues.clear();
    m_binULimits.clear();
    int size=  CalibrationXML::readAttribute<int>(dom,"size");
    
    DOMNode * n1 = dom->getFirstChild();
    int bin;
    for(bin=0; bin < size; bin ++)
    {
        while( ( n1->getNodeType() != DOMNode::ELEMENT_NODE ) && ( n1 != 0 ) )   n1 = n1->getNextSibling();
          if (n1)
	  {
	      DOMElement * binElement = (DOMElement *) n1;
              m_binValues.push_back(CalibrationXML::readAttribute<double>(binElement,"value"));
	
	      if(bin>0) m_normalization+=m_binValues[bin];
              m_binULimits.push_back(CalibrationXML::readAttribute<double>(binElement,"uLimit"));      
	       n1 = n1->getNextSibling();
	  }             
    }
    if(bin>0)
      m_binValues.push_back(CalibrationXML::readAttribute<int>(dom,"overFlowValue"));


  }
  
void  CalibratedHistogramXML::write (XERCES_CPP_NAMESPACE::DOMElement * dom) const
  {

    int size=m_binULimits.size();
    CalibrationXML::writeAttribute(dom,"size",size);
    DOMElement * binElement;
    for(int bin=0; bin < size; bin ++)
    {
       binElement = CalibrationXML::addChild(dom,"Bin");
       CalibrationXML::writeAttribute(binElement,"value",m_binValues[bin]);
       CalibrationXML::writeAttribute(binElement,"uLimit",m_binULimits[bin]);
    }
    CalibrationXML::writeAttribute(dom,"overFlowValue",m_binValues[size]);

  }

