#ifndef MESetNonObject_H
#define MESetNonObject_H

#include "MESet.h"

namespace ecaldqm
{
  class MESetNonObject : public MESet
  {
  public :
    MESetNonObject(std::string const&, MEData const&, bool _readOnly = false);
    ~MESetNonObject();

    void book();
    bool retrieve() const;

    void fill(double, double _wy = 1., double _w = 1.);
  };
}


#endif
