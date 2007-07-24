#ifndef DIGIECAL_EBDATAFRAME_H
#define DIGIECAL_EBDATAFRAME_H

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDigi/interface/EcalDataFrame.h"
#include <iosfwd>



/** \class EBDataFrame
      
$Id: EBDataFrame.h,v 1.5 2007/07/24 10:21:04 innocent Exp $
*/
class EBDataFrame : public EcalDataFrame 
{
 public:
  typedef EBDetId key_type; ///< For the sorted collection
  typedef EcalDataFrame Base;

  EBDataFrame() {}
  // EBDataFrame(DetId i) :  Base(i) {}
  EBDataFrame(edm::DataFrame const & base) : Base(base) {}
  EBDataFrame(EcalDataFrame const & base) : Base(base) {}
    
  virtual ~EBDataFrame() {}

  key_type id() const { return Base::id(); }

};

std::ostream& operator<<(std::ostream&, const EBDataFrame&);


#endif
