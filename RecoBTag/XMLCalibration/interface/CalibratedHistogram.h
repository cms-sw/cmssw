#ifndef CalibratedHistogram_H
#define CalibratedHistogram_H
#include <xercesc/dom/DOM.hpp>
#include "BReco/BTagCalibration/interface/CalibratedObject.h"
#include <vector>
#include <xercesc/dom/DOMNode.hpp>


/**
* This class implements some methods of the CalibratedObject.
* This class does not provide methdos for calibration, i.e.
*  [start|update|finish]Calibration() functions.
* If you want to use it in a calibration program you have to 
* implement those methods in a child class.
*/
using namespace std;
class CalibratedHistogram:public CalibratedObject
{
public:
  CalibratedHistogram() {} 
  virtual ~CalibratedHistogram() {} 
  CalibratedHistogram(vector<double> binLimits) : m_binULimits(binLimits),m_normalization(0) 
  { }

   
  void read (XERCES_CPP_NAMESPACE::DOMElement * dom);
  
  void write (XERCES_CPP_NAMESPACE::DOMElement * dom) const;

  std::string name () const
  {
    return "CalibratedHistogram";
  }

   /*
    zero is underflow, size is overflow (the number of good bins is size-1))
   */
  double binContent(int bin) const {return m_binValues[bin];}
  double value(double x) const {return binContent(findBin(x));}
  void  reset();
  void  setBinContent(int b,double v);
  void  fill(double x, double w=1.) 
  { 
   int bin=findBin(x);
   setBinContent(bin,binContent(bin)+w);
  }
  
  int findBin(double x) const ;

  double integral(double hBound, double lBound=0,int mode=0) const ;
  double normalizedIntegral(double hBound, double lBound=0,int mode=0) const
  {
   return integral(hBound,lBound,mode)/m_normalization;
  } 
  
protected:
  vector<double> m_binULimits;
  vector<double> m_binValues;
  double m_normalization;
};

#endif
