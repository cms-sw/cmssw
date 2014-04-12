// Class removed until concurrency issue is finalized
#if 0

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
    MESetChannel(std::string const&, binning::ObjectType, binning::BinningType, MonitorElement::Kind);
    MESetChannel(MESetChannel const&);
    ~MESetChannel();

    MESet& operator=(MESet const& _rhs) override;

    MESet* clone(std::string const& = "") const override;

    void book(DQMStore&) override;
    void book(DQMStore::IBooker&) override;
    bool retrieve(DQMStore const&, std::string* = 0) const override;
    void clear() const override;

    void fill(DetId const&, double = 1., double = 0., double = 0.) override;
    void fill(EcalElectronicsId const&, double = 1., double = 0., double = 0.) override;

    void setBinContent(DetId const&, double) override;
    void setBinContent(EcalElectronicsId const&, double) override;

    void setBinError(DetId const&, double) override;
    void setBinError(EcalElectronicsId const&, double) override;

    void setBinEntries(DetId const&, double) override;
    void setBinEntries(EcalElectronicsId const&, double) override;

    double getBinContent(DetId const&, int = 0) const override;
    double getBinContent(EcalElectronicsId const&, int = 0) const override;

    double getBinError(DetId const&, int = 0) const override;
    double getBinError(EcalElectronicsId const&, int = 0) const override;
    
    double getBinEntries(DetId const&, int = 0) const override;
    double getBinEntries(EcalElectronicsId const&, int = 0) const override;

    void reset(double = 0., double = 0., double = 0.) override;

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

#endif
