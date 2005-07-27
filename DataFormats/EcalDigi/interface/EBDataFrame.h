#ifndef DIGIECAL_EBDATAFRAME_H
#define DIGIECAL_EBDATAFRAME_H

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDigi/interface/EcalMGPASample.h"
#include <vector>
#include <ostream>

namespace cms {

  /** \class EBDataFrame
      
  $Id : $
  */

  class EBDataFrame {
  public:
    EBDataFrame(); // for persistence
    explicit EBDataFrame(const EBDetId& id);
    
    const EBDetId& id() const { return id_; }
    
    int size() const { return size_; }

    const EcalMGPASample& operator[](int i) const { return data_[i]; }
    const EcalMGPASample& sample(int i) const { return data_[i]; }
    
    void setSize(int size);
    //    void setPresamples(int ps);
    void setSample(int i, const EcalMGPASample& sam) { data_[i]=sam; }

    static const int MAXSAMPLES = 10;

  private:
    EBDetId id_;
    int size_;
    //    int ecalPresamples_;
    std::vector<EcalMGPASample> data_;    
  };
  
  std::ostream& operator<<(std::ostream&, const EBDataFrame&);
}



#endif
