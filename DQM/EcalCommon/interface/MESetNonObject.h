#ifndef MESetNonObject_H
#define MESetNonObject_H

#include "MESet.h"

namespace ecaldqm
{
  class MESetNonObject : public MESet
  {
  public :
    MESetNonObject(MEData const&);
    ~MESetNonObject();

    void book();
    bool retrieve() const;

    void fill(double, double _wy = 1., double _w = 1.);

    void setBinContent(int, double);

    void setBinError(int, double);

    void setBinEntries(int, double);

    double getBinContent(int) const;

    double getBinError(int) const;

    double getBinEntries(int) const;

    int findBin(double, double _y = 0.) const;
  };
}


#endif
