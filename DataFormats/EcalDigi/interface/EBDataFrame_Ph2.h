#ifndef DIGIECAL_EBDATAFRAME_PH2_H
#define DIGIECAL_EBDATAFRAME_PH2_H

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDigi/interface/EcalDataFrame_Ph2.h"
#include <iosfwd>

/** \class EBDataFrame
      
*/
class EBDataFrame_Ph2 : public EcalDataFrame_Ph2 {
public:
  typedef EBDetId key_type;
  typedef EcalDataFrame_Ph2 Base;

  EBDataFrame_Ph2() {}

  EBDataFrame_Ph2(edm::DataFrame const& base) : Base(base) {}
  EBDataFrame_Ph2(EcalDataFrame_Ph2 const& base) : Base(base) {}

  ~EBDataFrame_Ph2() override {}

  key_type id() const { return Base::id(); }
};

std::ostream& operator<<(std::ostream&, const EBDataFrame_Ph2&);

#endif
