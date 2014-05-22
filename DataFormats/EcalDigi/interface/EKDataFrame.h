#ifndef DIGIECAL_EKDATAFRAME_H
#define DIGIECAL_EKDATAFRAME_H

#include "DataFormats/EcalDetId/interface/EKDetId.h"
#include "DataFormats/EcalDigi/interface/EcalDataFrame.h"
#include <iosfwd>



/** \class EKDataFrame
      
$Id: EKDataFrame.h,v 1.6 2014/04/02 10:57:51 shervin Exp $
*/


class EKDataFrame : public EcalDataFrame 
{
 public:
  typedef EKDetId key_type; ///< For the sorted collection
  typedef EcalDataFrame Base;

  EKDataFrame() {}
  // EKDataFrame(DetId i) :  Base(i) {}
  EKDataFrame(edm::DataFrame const & base) : Base(base) {}
  EKDataFrame(EcalDataFrame const & base) : Base(base) {}
    
  virtual ~EKDataFrame() {}

  key_type id() const { return Base::id(); }

};


std::ostream& operator<<(std::ostream&, const EKDataFrame&);


#endif
