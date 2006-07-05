#ifndef DIGIECAL_EBDATAFRAME_H
#define DIGIECAL_EBDATAFRAME_H

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDigi/interface/EcalDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalMGPASample.h"




/** \class EBDataFrame
      
$Id: EBDataFrame.h,v 1.3 2006/06/24 13:28:11 meridian Exp $
*/

class EBDataFrame : public EcalDataFrame 
{
 public:
  typedef EBDetId key_type; ///< For the sorted collection

  EBDataFrame(); // for persistence
  explicit EBDataFrame(const EBDetId& id);
    
  virtual ~EBDataFrame() {};

  virtual const EBDetId& id() const { return id_; }

 private:
  EBDetId id_;

};
  
std::ostream& operator<<(std::ostream&, const EBDataFrame&);



#endif
