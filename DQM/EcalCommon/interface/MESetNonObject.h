#ifndef MESetNonObject_H
#define MESetNonObject_H

#include "MESet.h"

namespace ecaldqm
{
  class MESetNonObject : public MESet
  {
  public :
    MESetNonObject(std::string const&, binning::ObjectType, binning::BinningType, MonitorElement::Kind, binning::AxisSpecs const* = 0, binning::AxisSpecs const* = 0, binning::AxisSpecs const* = 0);
    MESetNonObject(MESetNonObject const&);
    ~MESetNonObject();

    MESet& operator=(MESet const&) override;

    MESet* clone(std::string const& = "") const override;

    void book(DQMStore&) override;
    void book(DQMStore::IBooker&) override;
    bool retrieve(DQMStore const&, std::string* = 0) const override;

    void fill(double, double = 1., double = 1.) override;

    void setBinContent(int, double) override;

    void setBinError(int, double) override;

    void setBinEntries(int, double) override;

    double getBinContent(int, int = 0) const override;

    double getBinError(int, int = 0) const override;

    double getBinEntries(int, int = 0) const override;

    int findBin(double, double = 0.) const;

    bool isVariableBinning() const override;

  protected:
    binning::AxisSpecs const* xaxis_;
    binning::AxisSpecs const* yaxis_;
    binning::AxisSpecs const* zaxis_;

  private:
    template<class Bookable> void doBook_(Bookable&);
  };
}


#endif
