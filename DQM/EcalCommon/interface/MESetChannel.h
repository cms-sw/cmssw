#ifndef MESetChannel_H
#define MESetChannel_H

#include <map>

#include "MESet.h"

namespace ecaldqm
{
  class MESetChannel : public MESet
  {
  public :
    MESetChannel(std::string const&, MEData const&, bool _readOnly = false);
    ~MESetChannel();

    bool retrieve() const;
    void clear() const;

    void fill(DetId const&, float _w = 1., float _unused1 = 0., float _unused2 = 0.);
    void fill(EcalElectronicsId const&, float _w = 1., float _unused1 = 0., float _unused2 = 0.);
    //    void fill(int, float _w = 1., float _unused1 = 0., float _unused2 = 0.);

    void setBinContent(DetId const&, float, float _err = 0.);
    void setBinContent(EcalElectronicsId const&, float, float _err = 0.);

    void reset(float _content = 0., float _err = 0., float _entries = 0.);

    float getBinContent(DetId const&, int _bin = 0) const;
    float getBinContent(EcalElectronicsId const&, int _bin = 0) const;
    
    float getBinEntries(DetId const& _id, int _bin = 0) const { return getBinContent(_id, _bin); }
    float getBinEntries(EcalElectronicsId const& _id, int _bin = 0) const { return getBinContent(_id, _bin); }

    void checkDirectory() const;

  private :
    std::map<uint32_t, unsigned>::iterator append_(std::string const&, uint32_t);
    uint32_t getIndex_(DetId const&) const;
    uint32_t getIndex_(EcalElectronicsId const&) const;

    mutable std::vector<MonitorElement*> mes_;
    mutable std::map<uint32_t, unsigned> meTable_;

    // have readmode param for MESet and a parameter in the Ctor
    mutable bool readMode_;
  };
}

#endif
