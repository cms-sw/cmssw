#ifndef DIGIECAL_EEDATAFRAME_H
#define DIGIECAL_EEDATAFRAME_H

#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDigi/interface/EcalDataFrame.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include <iosfwd>



/** \class EEDataFrame
      
$Id: EEDataFrame.h,v 1.4 2006/07/05 17:38:51 meridian Exp $
*/


class EEDataFrame : public EcalDataFrame 
{
 public:
  typedef EEDetId key_type; ///< For the sorted collection
  typedef EcalDataFrame Base;

  EEDataFrame() {}
  // EEDataFrame(DetId i) :  Base(i) {}
  EEDataFrame(DataFrame const & base) : Base(base) {}
  EEDataFrame(EcalDataFrame const & base) : Base(base) {}
    
  virtual ~EEDataFrame() {}

  key_type id() const { return Base::id(); }

};


std::ostream& operator<<(std::ostream&, const EEDataFrame&);


#endif
