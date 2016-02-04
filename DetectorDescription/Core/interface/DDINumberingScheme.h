#ifndef DDD_DDINumberingScheme
#define DDD_DDINumberingScheme

#include "DetectorDescription/Core/interface/DDExpandedNode.h"

/** abstract interface for a numbering scheme */
class DDINumberingScheme
{
public:
  virtual int id(const DDExpandedNode &) const = 0;
  virtual int id(const DDGeoHistory &) const = 0;
  virtual DDExpandedNode node(int) const  = 0;
  virtual DDGeoHistory history(int) const = 0;
};

#endif // DDD_DDINumberingScheme
