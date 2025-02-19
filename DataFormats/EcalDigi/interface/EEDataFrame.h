#ifndef DIGIECAL_EEDATAFRAME_H
#define DIGIECAL_EEDATAFRAME_H

#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDigi/interface/EcalDataFrame.h"
#include <iosfwd>



/** \class EEDataFrame
      
$Id: EEDataFrame.h,v 1.6 2007/07/24 10:57:51 innocent Exp $
*/


class EEDataFrame : public EcalDataFrame 
{
 public:
  typedef EEDetId key_type; ///< For the sorted collection
  typedef EcalDataFrame Base;

  EEDataFrame() {}
  // EEDataFrame(DetId i) :  Base(i) {}
  EEDataFrame(edm::DataFrame const & base) : Base(base) {}
  EEDataFrame(EcalDataFrame const & base) : Base(base) {}
    
  virtual ~EEDataFrame() {}

  key_type id() const { return Base::id(); }

};


std::ostream& operator<<(std::ostream&, const EEDataFrame&);


#endif
