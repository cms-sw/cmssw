#ifndef DIGIHCAL_HFDATAFRAME_H
#define DIGIHCAL_HFDATAFRAME_H

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"
#include "DataFormats/HcalDigi/interface/HcalQIESample.h"
#include <vector>
#include <ostream>

namespace cms {

  /** \class cms::HODataFrame
       
  Precision readout digi for HO
  
   */
  class HFDataFrame {
  public:
    HFDataFrame(); // for persistence
    explicit HFDataFrame(const HcalDetId& id);
    
    const HcalDetId& id() const { return id_; }
    const HcalElectronicsId& elecId() const { return electronicsId_; }
    
    /// total number of samples in the digi
    int size() const { return size_; }
    /// number of samples before the sample from the triggered beam crossing (according to the hardware)
    int presamples() const { return hcalPresamples_; }
    
    /// access a sample
    const HcalQIESample& operator[](int i) const { return data_[i]; }
    /// access a sample
    const HcalQIESample& sample(int i) const { return data_[i]; }
   
    /// validate appropriate DV and ER bits as well as capid rotation for the specified samples (default is all)
    bool validate(int firstSample=0, int nSamples=100) const;
    
    void setSize(int size);
    void setPresamples(int ps);
    void setSample(int i, const HcalQIESample& sam) { data_[i]=sam; }
    void setReadoutIds(const HcalElectronicsId& eid);
    
    static const int MAXSAMPLES = 10;
  private:
    cms::HcalDetId id_;
    cms::HcalElectronicsId electronicsId_; 
    int size_;
    int hcalPresamples_;
    std::vector<cms::HcalQIESample> data_;
  };

  std::ostream& operator<<(std::ostream&, const HFDataFrame&);

}




#endif
