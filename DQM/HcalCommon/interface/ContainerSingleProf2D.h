#ifndef ContainerSingleProf2D_h
#define ContainerSingleProf2D_h

/*
 *	file:			ContainerSignle2D.h
 *	Author:			Viktor Khristenko
 *
 *	Description:
 *		Container to hold a single ME - for convenience of initialization
 */

#include "DQM/HcalCommon/interface/ContainerSingle2D.h"

#include <string>

namespace hcaldqm
{
  class ContainerSingleProf2D : public ContainerSingle2D
  {
  public:
    ContainerSingleProf2D();
    ContainerSingleProf2D(std::string const& folder,
                          quantity::Quantity*, quantity::Quantity*,
                          quantity::Quantity *qz = new quantity::ValueQuantity(quantity::fN));
    ~ContainerSingleProf2D() override {}

    void initialize(std::string const& folder,
                    quantity::Quantity*, quantity::Quantity*,
                    quantity::Quantity *qz = new quantity::ValueQuantity(quantity::fN),
                    int debug=0) override;

    void initialize(std::string const& folder,
                    std::string const&,
                    quantity::Quantity*, quantity::Quantity*,
                    quantity::Quantity *qz = new quantity::ValueQuantity(quantity::fN),
                    int debug=0) override;

    //	booking
    void book(DQMStore::IBooker&,
              std::string subsystem="Hcal", std::string aux="") override;

    void fill(int, int) override;
    void fill(int, double) override;
    void fill(int, double, double) override;
    void fill(int, int, int) override;
    void fill(int, int, double) override;
    void fill(double, int) override;
    void fill(double, double) override;
    void fill(double, double, double) override;

    void fill(HcalDetId const&) override;
    void fill(HcalDetId const&, int) override;
    void fill(HcalDetId const&, double) override;
    void fill(HcalDetId const&, int, int) override;
    void fill(HcalDetId const&, int, double) override;
    void fill(HcalDetId const&, double, double) override;

    void fill(HcalElectronicsId const&) override;
    void fill(HcalElectronicsId const&, int) override;
    void fill(HcalElectronicsId const&, double) override;
    void fill(HcalElectronicsId const&, int, int) override;
    void fill(HcalElectronicsId const&, int, double) override;
    void fill(HcalElectronicsId const&, double, double) override;

    void fill(HcalDetId const&, HcalElectronicsId const&) override;
    void fill(HcalDetId const&, HcalElectronicsId const&,
              double) override;

    void fill(HcalTrigTowerDetId const&) override;
    void fill(HcalTrigTowerDetId const&, int) override;
    void fill(HcalTrigTowerDetId const&, double) override;
    void fill(HcalTrigTowerDetId const&, int, int) override;
    void fill(HcalTrigTowerDetId const&, int, double) override;
    void fill(HcalTrigTowerDetId const&, double, double) override;
  };
}

#endif
