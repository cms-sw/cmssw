#ifndef DIGIECAL_ECALDATAFRAME_H
#define DIGIECAL_ECALDATAFRAME_H

#include "DataFormats/EcalDigi/interface/EcalMGPASample.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Common/interface/DataFrame.h"




/** \class EcalDataFrame
      
$Id: EcalDataFrame.h,v 1.6 2007/07/25 06:58:15 innocent Exp $
*/
class EcalDataFrame {
 public:
  EcalDataFrame() {}
  // EcalDataFrame(DetId i) :  m_data(i) {}
  EcalDataFrame(edm::DataFrame const & iframe) : m_data(iframe){} 

  virtual ~EcalDataFrame() {} 

  DetId id() const { return m_data.id();}
    
  int size() const { return m_data.size();}

  EcalMGPASample operator[](int i) const { return m_data[i];}
  EcalMGPASample sample(int i) const { return m_data[i]; }
    
  // FIXME (shall we throw??)
  void setSize(int){}
  // void setPresamples(int ps);
  void setSample(int i, EcalMGPASample sam) { m_data[i]=sam; }

  static const int MAXSAMPLES = 10;

  edm::DataFrame const & frame() const { return m_data;}
  edm::DataFrame & frame() { return m_data;}

 private:
 
  edm::DataFrame m_data;
  
};
  
#endif
