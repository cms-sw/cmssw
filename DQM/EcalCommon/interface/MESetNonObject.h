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

    void fill(float, float _wy = 1., float _w = 1.);
  };
}


#endif
