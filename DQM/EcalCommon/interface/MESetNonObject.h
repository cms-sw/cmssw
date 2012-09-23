#ifndef MESetNonObject_H
#define MESetNonObject_H

#include "MESet.h"

namespace ecaldqm
{
  class MESetNonObject : public MESet
  {
  public :
    MESetNonObject(std::string const&, BinService::ObjectType, BinService::BinningType, MonitorElement::Kind, BinService::AxisSpecs const* = 0, BinService::AxisSpecs const* = 0, BinService::AxisSpecs const* = 0);
    MESetNonObject(MESetNonObject const&);
    ~MESetNonObject();

    MESet& operator=(MESet const&);

    MESet* clone() const;

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

    bool isVariableBinning() const;

  protected:
    BinService::AxisSpecs const* xaxis_;
    BinService::AxisSpecs const* yaxis_;
    BinService::AxisSpecs const* zaxis_;
  };
}


#endif
