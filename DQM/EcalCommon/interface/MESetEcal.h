#ifndef MESetEcal_H
#define MESetEcal_H

#include "MESet.h"

namespace ecaldqm
{

  class MESetEcal : public MESet
  {
  public :
    MESetEcal(std::string const&, MEData const&, int logicalDimensions_, bool _readOnly = false);
    ~MESetEcal();

    void book();
    bool retrieve() const;

    void fill(DetId const&, double _wx = 1., double _wy = 1., double _w = 1.);
    void fill(unsigned, double _wx = 1., double _wy = 1., double _w = 1.);
    void fill(double, double _wy = 1., double _w = 1.);

    void setBinContent(DetId const&, double, double _err = 0.);
    void setBinContent(unsigned, double, double _err = 0.);

    void setBinEntries(DetId const&, double);
    void setBinEntries(unsigned, double);

    double getBinContent(DetId const&, int _bin = 0) const;
    double getBinContent(unsigned, int _bin = 0) const;

    double getBinError(DetId const&, int _bin = 0) const;
    double getBinError(unsigned, int _bin = 0) const;

    double getBinEntries(DetId const&, int _bin = 0) const;
    double getBinEntries(unsigned, int _bin = 0) const;

    void reset(double _content = 0., double _err = 0., double _entries = 0.);

    std::vector<std::string> generateNames() const;

  protected :
    virtual void find_(uint32_t) const;
    virtual void fill_(double); // method for derived classes

    const unsigned logicalDimensions_;

    mutable uint32_t cacheId_;
    mutable std::pair<unsigned, std::vector<int> > cache_;
  };

}

#endif
