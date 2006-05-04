#include "RecoBTag/XMLCalibration/interface/CalibratedHistogram.h"
#include "RecoBTag/XMLCalibration/interface/CalibrationXML.h"
#include <iostream>

using namespace std;
void  CalibratedHistogram::read (XERCES_CPP_NAMESPACE::DOMElement * dom)
  {
    m_normalization=0; 
    m_binValues.clear();
    m_binULimits.clear();
    int size=  CalibrationXML::readAttribute<int>(dom,"size");
    cout << "size " << size << endl; 
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
  
void  CalibratedHistogram::write (XERCES_CPP_NAMESPACE::DOMElement * dom) const
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

void CalibratedHistogram::reset()
{
    int size=m_binValues.size();
    for(int bin=0; bin < size; bin ++)
    {
      setBinContent(bin,0);
    }
}

void CalibratedHistogram::setBinContent(int bin,double value) 
   { 
    if(m_binULimits.size()+1 != m_binValues.size()) 
         m_binValues.resize(m_binULimits.size()+1);

     if(bin>0) { 
        if(bin < m_binULimits.size()) {
          
	 m_normalization+=value-m_binValues[bin]; // integral is only for good bin (no over/under flow)
	 m_binValues[bin]=value;
        }
	else {
	 m_binValues[m_binULimits.size()]=value;
	}
      }
      else {
	 m_binValues[0]=value;
      }
}
  
int CalibratedHistogram::findBin(double x) const
{
 int bin;
 int size = m_binULimits.size();
 for(bin=0;bin< size ; bin ++)
 {
  if(m_binULimits[bin] > x) return bin;
 }
 return bin;
}

double CalibratedHistogram::integral(double hBound, double lBound,int mode) const
  {
   int lBin=findBin(lBound);
   int hBin=findBin(hBound);
   double sum=0;

   for(int bin=lBin+1;bin<hBin;bin++)
   {
     sum+=m_binValues[bin];
   //  cout << "+"<< m_binValues[bin];
   }
//   cout << "sum="<<sum << endl;
   if(0)  //TODO: mode = linear VS mode = high bound / low bound
    {
     if(hBin-1>0)
     {
       double hSlope=(m_binValues[hBin]-m_binValues[hBin-1])/(m_binULimits[hBin]-m_binULimits[hBin-1]);
       double deltaX=hBound-m_binULimits[hBin-1];
       cout << "High bound interpolation " << hSlope << "*" << deltaX << " = " << hSlope*deltaX << endl;
       sum+=hSlope*deltaX;   
     }
     if(lBin-1>0)
     {
       double hSlope=(m_binValues[lBin]-m_binValues[lBin-1])/(m_binULimits[lBin]-m_binULimits[lBin-1]);
       double deltaX=m_binULimits[lBin]-lBound;
       cout << "Low bound interpolation " << hSlope << "*" << deltaX << " = " << hSlope*deltaX << endl;
       sum+=hSlope*deltaX;   
     }
    }
    
   return sum;
  }  
