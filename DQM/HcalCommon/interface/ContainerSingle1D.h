#ifndef ContainerSingle1D_h
#define ContainerSingle1D_h

/*
 *      file:                   ContainerSignle1D.h
 *      Author:                 Viktor Khristenko
 *
 *      Description:
 *              Container to hold a single ME - for convenience of initialization
 */

#include "DQM/HcalCommon/interface/Container.h"
#include "DQM/HcalCommon/interface/ValueQuantity.h"

#include <string>

namespace hcaldqm
{
  class ContainerSingle1D : public Container
  {
  public:
    ContainerSingle1D();
    ContainerSingle1D(std::string const& folder,
                      quantity::Quantity*,
                      quantity::Quantity *qy = new quantity::ValueQuantity(quantity::fN));
    ContainerSingle1D(std::string const& folder,
                      std::string const&,
                      quantity::Quantity*,
                      quantity::Quantity *qy = new quantity::ValueQuantity(quantity::fN));
    ContainerSingle1D(ContainerSingle1D const&);
    ~ContainerSingle1D() override;

    virtual void initialize(std::string const& folder,
                            quantity::Quantity*,
                            quantity::Quantity *qy = new quantity::ValueQuantity(quantity::fN),
                            int debug=0);

    virtual void initialize(std::string const& folder,
                            std::string const&,
                            quantity::Quantity*,
                            quantity::Quantity *qy = new quantity::ValueQuantity(quantity::fN),
                            int debug=0);
    using Container::initialize;
    //  booking
    virtual void book(DQMStore::IBooker&,
                      std::string subsystem="Hcal", std::string aux="");
    //  filling
    virtual void fill(int);
    virtual void fill(double);
    virtual void fill(int, int);
    virtual void fill(int, double);
    virtual void fill(double, int);
    virtual void fill(double, double);

    virtual double getBinContent(int) ;
    virtual double getBinContent(double);
    virtual double getBinEntries(int);
    virtual double getBinEntries(double);

    virtual void setBinContent(int, int);
    virtual void setBinContent(int, double);
    virtual void setBinContent(double, int);
    virtual void setBinContent(double, double);

    virtual void fill(HcalDetId const&);
    virtual void fill(HcalDetId const&, double);
    virtual void fill(HcalDetId const&, double, double);

    virtual double getBinContent(HcalDetId const&);
    virtual double getBinEntries(HcalDetId const&);

    virtual void setBinContent(HcalDetId const&, int);
    virtual void setBinContent(HcalDetId const&, double);

    virtual void fill(HcalElectronicsId const&);
    virtual void fill(HcalElectronicsId const&, double);
    virtual void fill(HcalElectronicsId const&, double, double);

    virtual double getBinContent(HcalElectronicsId const&);
    virtual double getBinEntries(HcalElectronicsId const&);

    virtual void setBinContent(HcalElectronicsId const&, int);
    virtual void setBinContent(HcalElectronicsId const&, double);

    virtual void fill(HcalTrigTowerDetId const&);
    virtual void fill(HcalTrigTowerDetId const&, double);
    virtual void fill(HcalTrigTowerDetId const&, double, double);

    virtual double getBinContent(HcalTrigTowerDetId const&);
    virtual double getBinEntries(HcalTrigTowerDetId const&);

    virtual void setBinContent(HcalTrigTowerDetId const&, int);
    virtual void setBinContent(HcalTrigTowerDetId const&, double);

    virtual void reset() {_me->Reset();}
    virtual void print() {std::cout << _qname << std::endl;}

    virtual void extendAxisRange(int);

    virtual void showOverflowX(bool showOverflow);
    virtual void showOverflowY(bool showOverflow);

  protected:
    MonitorElement*     _me;
    quantity::Quantity* _qx;
    quantity::Quantity* _qy;

    virtual void customize();
  };
}

#endif
