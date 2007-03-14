#ifndef CalibratedHistogram_H
#define CalibratedHistogram_H
#include <vector>
//#include<boost/cstdint.hpp>

class CalibratedHistogram //:public CalibratedObject
{
public:
  CalibratedHistogram() {} 
  virtual ~CalibratedHistogram() {} 
  CalibratedHistogram(std::vector<float> binLimits) : m_binULimits(binLimits),m_normalization(0) 
  { }

  std::vector<float> values() const { return m_binValues;}
  std::vector<float> upperLimits() const {return m_binULimits; }
  void  setUpperLimits(std::vector<float> limits) { m_binULimits=limits; }
  void  setValues(std::vector<float> values) { m_binValues=values; normalize(); }
  void normalize() {m_normalization=0; for(size_t i=1;i<m_binValues.size(); i++) m_normalization+=m_binValues[i];}   

  /*
    zero is underflow, size is overflow (the number of good bins is size-1))
  */
  float binContent(int bin) const {return m_binValues[bin];}
  float value(float x) const {return binContent(findBin(x));}
  void  reset();
  void  setBinContent(int b,float v);
  void  fill(float x, float w=1.) 
  { 
   int bin=findBin(x);
   setBinContent(bin,binContent(bin)+w);
  }
  
  int findBin(float x) const ;

  float integral(float hBound, float lBound=0,int mode=0) const ;
  float normalizedIntegral(float hBound, float lBound=0,int mode=0) const
  {
   return integral(hBound,lBound,mode)/m_normalization;
  } 
 
  float normalization() const { return m_normalization; }
   
protected:
  std::vector<float> m_binULimits;
  std::vector<float> m_binValues;
  float m_normalization;
};

#endif
