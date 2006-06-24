#ifndef DIGIECAL_EBDATAFRAME_H
#define DIGIECAL_EBDATAFRAME_H

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDigi/interface/EcalDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalMGPASample.h"




/** \class EBDataFrame
      
$Id: $
*/

class EBDataFrame : public EcalDataFrame 
{
 public:
  typedef EBDetId key_type; ///< For the sorted collection

  EBDataFrame(); // for persistence
  explicit EBDataFrame(const EBDetId& id);
    
  virtual ~EBDataFrame() {};

  virtual const DetId& id() const { return id_; }

 private:
  EBDetId id_;

};
  
std::ostream& operator<<(std::ostream&, const EBDataFrame&);



#endif
