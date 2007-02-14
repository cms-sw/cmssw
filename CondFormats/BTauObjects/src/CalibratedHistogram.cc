#include "CondFormats/BTauObjects/interface/CalibratedHistogram.h"
#include <iostream>

using namespace std;

void CalibratedHistogram::reset()
{
    int size=m_binValues.size();
    for(int bin=0; bin < size; bin ++)
    {
      setBinContent(bin,0);
    }
}

void CalibratedHistogram::setBinContent(int bin,float value) 
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
  
int CalibratedHistogram::findBin(float x) const
{
 int bin;
 int size = m_binULimits.size();
 for(bin=0;bin< size ; bin ++)
 {
  if(m_binULimits[bin] > x) return bin;
 }
 return bin;
}

float CalibratedHistogram::integral(float hBound, float lBound,int mode) const
  {
   int lBin=findBin(lBound);
   int hBin=findBin(hBound);
   float sum=0;

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
       float hSlope=(m_binValues[hBin]-m_binValues[hBin-1])/(m_binULimits[hBin]-m_binULimits[hBin-1]);
       float deltaX=hBound-m_binULimits[hBin-1];
       cout << "High bound interpolation " << hSlope << "*" << deltaX << " = " << hSlope*deltaX << endl;
       sum+=hSlope*deltaX;   
     }
     if(lBin-1>0)
     {
       float hSlope=(m_binValues[lBin]-m_binValues[lBin-1])/(m_binULimits[lBin]-m_binULimits[lBin-1]);
       float deltaX=m_binULimits[lBin]-lBound;
       cout << "Low bound interpolation " << hSlope << "*" << deltaX << " = " << hSlope*deltaX << endl;
       sum+=hSlope*deltaX;   
     }
    }
    
   return sum;
  }  
