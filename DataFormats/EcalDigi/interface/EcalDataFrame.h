#ifndef DIGIECAL_ECALDATAFRAME_H
#define DIGIECAL_ECALDATAFRAME_H

#include "DataFormats/EcalDigi/interface/EcalMGPASample.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Common/interface/DataFrame.h"




/** \class EcalDataFrame
      
$Id: EcalDataFrame.h,v 1.4 2007/07/24 10:21:04 innocent Exp $
*/
class EcalDataFrame {
 public:
  EcalDataFrame() {}
  // EcalDataFrame(DetId i) :  m_data(i) {}
  EcalDataFrame(edm::DataFrame const & iframe) : m_data(iframe){} 

  virtual ~EcalDataFrame() {} 

  DetId id() const { return m_data.id();}
    
  int size() const { return  MAXSAMPLES;}

  EcalMGPASample operator[](int i) const { return m_data[i];}
  EcalMGPASample sample(int i) const { return m_data[i]; }
    
  void setSize(int){}
  // void setPresamples(int ps);
  void setSample(int i, EcalMGPASample sam) { m_data[i]=sam; }

  static const int MAXSAMPLES = 10;

 private:
 
  edm::DataFrame m_data;
  
};
  
#endif
