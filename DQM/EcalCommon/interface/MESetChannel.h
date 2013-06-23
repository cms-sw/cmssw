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

    void fill(DetId const&, double _w = 1., double _unused1 = 0., double _unused2 = 0.);
    void fill(EcalElectronicsId const&, double _w = 1., double _unused1 = 0., double _unused2 = 0.);
    //    void fill(int, double _w = 1., double _unused1 = 0., double _unused2 = 0.);

    void setBinContent(DetId const&, double, double _err = 0.);
    void setBinContent(EcalElectronicsId const&, double, double _err = 0.);

    void reset(double _content = 0., double _err = 0., double _entries = 0.);

    double getBinContent(DetId const&, int _bin = 0) const;
    double getBinContent(EcalElectronicsId const&, int _bin = 0) const;
    
    double getBinEntries(DetId const& _id, int _bin = 0) const { return getBinContent(_id, _bin); }
    double getBinEntries(EcalElectronicsId const& _id, int _bin = 0) const { return getBinContent(_id, _bin); }

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
