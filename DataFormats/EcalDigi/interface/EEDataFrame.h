#ifndef DIGIECAL_EEDATAFRAME_H
#define DIGIECAL_EEDATAFRAME_H

#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDigi/interface/EcalDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalMGPASample.h"

/** \class EEDataFrame
      
$Id : $
*/

class EEDataFrame : public EcalDataFrame
{
 public:
  typedef EEDetId key_type; ///< For the sorted collection

  EEDataFrame(); // for persistence
  explicit EEDataFrame(const EEDetId& id);
    
  virtual ~EEDataFrame() {};

  virtual const EEDetId& id() const { return id_; }
    
 private:
  EEDetId id_;

};
  

std::ostream& operator<<(std::ostream&, const EEDataFrame&);




#endif
