#ifndef CondFormats_CSCObjects_CSCL1TPLookupTableCCLUT_h
#define CondFormats_CSCObjects_CSCL1TPLookupTableCCLUT_h

#include "CondFormats/Serialization/interface/Serializable.h"
#include <vector>
#include <unordered_map>

class CSCL1TPLookupTableCCLUT {
public:
  CSCL1TPLookupTableCCLUT();
  ~CSCL1TPLookupTableCCLUT() {}

  typedef std::unordered_map<unsigned, std::vector<unsigned> > t_lut;

  // setters
  void set_cclutPosition(t_lut lut);
  void set_cclutSlope(t_lut lut);

  // getters
  unsigned cclutPosition(unsigned pattern, unsigned code) const;
  unsigned cclutSlope(unsigned pattern, unsigned code) const;

private:
  t_lut cclutPosition_;
  t_lut cclutSlope_;

  COND_SERIALIZABLE;
};

#endif
