#ifndef CalibratedHistogram_H
#define CalibratedHistogram_H
#include <vector>
//#include<boost/cstdint.hpp>

class CalibratedHistogram //:public CalibratedObject
{
public:
  static std::vector<float>  constantBinning(int nBins, float min, float max)
   {
    std::vector<float> binULimits;
    float step=(max-min)/nBins;
    for(int i=0;i<nBins;i++)
    {
       binULimits.push_back(min+step*i);
    }
    binULimits.push_back(max); // last bin can be larger due to step approx

  return binULimits;
  }
  CalibratedHistogram() {} 
  virtual ~CalibratedHistogram() {} 
  CalibratedHistogram(std::vector<float> binLimits) : m_binULimits(binLimits),m_normalization(0) 
  { }
  CalibratedHistogram(int nBins, float min, float max) : m_binULimits(constantBinning(nBins,min,max)),m_normalization(0)
  {
    m_binValues.resize(nBins+1);
  }

  std::vector<float> values() const { return m_binValues;}
  std::vector<float> upperLimits() const {return m_binULimits; }

  ///Set bin limits (values can be meaningless if reset() or setValues() are not called )
  void  setUpperLimits(std::vector<float> limits) { m_binULimits=limits; }

  ///Set values at once from a vector
  void  setValues(std::vector<float> values) { m_binValues=values; normalize(); }

  //recompute normalization
  void normalize() {m_normalization=0; for(size_t i=1;i<m_binValues.size(); i++) m_normalization+=m_binValues[i];}   

  /*
    zero is underflow,1 is first bin,nbins is last bin,  nbins+1 is overflow bin 
  */
  float binContent(int bin) const {return m_binValues[bin];}

  ///return the number of bins
  int numberOfBins()const {return m_binULimits.size()-1; }

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
