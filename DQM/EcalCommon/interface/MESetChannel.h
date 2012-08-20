#ifndef MESetChannel_H
#define MESetChannel_H

#include <map>

#include "MESet.h"

namespace ecaldqm
{

  /* class MESetChannel
     MonitorElement wrapper for single-bin plots (TH1F, TProfile)
     One-to-one correspondence between id <-> histogram
  */

  class MESetChannel : public MESet
  {
  public :
    MESetChannel(MEData const&);
    ~MESetChannel();

    void book();
    bool retrieve() const;
    void clear() const;

    void fill(DetId const&, double _w = 1., double _unused1 = 0., double _unused2 = 0.);
    void fill(EcalElectronicsId const&, double _w = 1., double _unused1 = 0., double _unused2 = 0.);
    void fill(unsigned, double _w = 1., double _unused1 = 0., double _unused2 = 1.);

    void setBinContent(DetId const&, double);
    void setBinContent(EcalElectronicsId const&, double);
    void setBinContent(unsigned, double);

    void setBinError(DetId const&, double);
    void setBinError(EcalElectronicsId const&, double);
    void setBinError(unsigned, double);

    void setBinEntries(DetId const& _id, double);
    void setBinEntries(EcalElectronicsId const& _id, double);
    void setBinEntries(unsigned _dcctccid, double);

    double getBinContent(DetId const&, int _unused = 0) const;
    double getBinContent(EcalElectronicsId const&, int _unused = 0) const;
    double getBinContent(unsigned, int _unused = 0) const;

    double getBinError(DetId const&, int _unused = 0) const;
    double getBinError(EcalElectronicsId const&, int _unused = 0) const;
    double getBinError(unsigned, int _unused = 0) const;
    
    double getBinEntries(DetId const& _id, int _unused = 0) const;
    double getBinEntries(EcalElectronicsId const& _id, int _unused = 0) const;
    double getBinEntries(unsigned _dcctccid, int _unused = 0) const;

    void reset(double _content = 0., double _err = 0., double _entries = 0.);

    void checkDirectory() const;

  private :
    unsigned preparePlot_(uint32_t) const;
    unsigned findPlot_(uint32_t) const;
    uint32_t getIndex_(DetId const&) const;
    uint32_t getIndex_(EcalElectronicsId const&) const;

    mutable std::map<uint32_t, unsigned> meTable_;
  };
}

#endif
