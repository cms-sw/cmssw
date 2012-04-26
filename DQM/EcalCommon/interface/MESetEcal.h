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

    void fill(DetId const&, float _wx = 1., float _wy = 1., float _w = 1.);
    void fill(unsigned, float _wx = 1., float _wy = 1., float _w = 1.);
    void fill(float, float _wy = 1., float _w = 1.);

    void setBinContent(DetId const&, float, float _err = 0.);
    void setBinContent(unsigned, float, float _err = 0.);

    void setBinEntries(DetId const&, float);
    void setBinEntries(unsigned, float);

    float getBinContent(DetId const&, int _bin = 0) const;
    float getBinContent(unsigned, int _bin = 0) const;

    float getBinError(DetId const&, int _bin = 0) const;
    float getBinError(unsigned, int _bin = 0) const;

    float getBinEntries(DetId const&, int _bin = 0) const;
    float getBinEntries(unsigned, int _bin = 0) const;

    void reset(float _content = 0., float _err = 0., float _entries = 0.);

    std::vector<std::string> generateNames() const;

  protected :
    virtual void find_(uint32_t) const;
    virtual void fill_(float); // method for derived classes
    virtual void fill_(unsigned, float, float, float);

    const unsigned logicalDimensions_;

    mutable uint32_t cacheId_;
    mutable std::pair<unsigned, std::vector<int> > cache_;
  };

}

#endif
