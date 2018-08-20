#ifndef ContainerProf2D_h
#define ContainerProf2D_h

/*
 *      file:           ContainerProf2D.h
 *      Author:         Viktor Khristenko
 *
 *      Description:
 *              Container to hold TProfile or like
 *
 */

#include "DQM/HcalCommon/interface/Container2D.h"

#include <vector>
#include <string>

namespace hcaldqm
{
  class ContainerProf2D : public Container2D
  {
  public:
    ContainerProf2D();
    ContainerProf2D(std::string const& folder,
                    hashfunctions::HashType,
                    quantity::Quantity*, quantity::Quantity*,
                    quantity::Quantity* qz = new quantity::ValueQuantity(quantity::fEnergy));
    ~ContainerProf2D() override {}

    void initialize(std::string const& folder,
                    hashfunctions::HashType,
                    quantity::Quantity*, quantity::Quantity*,
                    quantity::Quantity *qz = new quantity::ValueQuantity(quantity::fEnergy),
                    int debug=0) override;

    void initialize(std::string const& folder,
                    std::string const& qname,
                    hashfunctions::HashType,
                    quantity::Quantity*, quantity::Quantity*,
                    quantity::Quantity *qz = new quantity::ValueQuantity(quantity::fEnergy),
                    int debug=0) override;

    void book(DQMStore::IBooker&,
              HcalElectronicsMap const*,
              std::string subsystem="Hcal", std::string aux="") override;
    void book(DQMStore::IBooker&,
              HcalElectronicsMap const*, filter::HashFilter const&,
              std::string subsystem="Hcal", std::string aux="") override;

    void fill(HcalDetId const&) override ;
    void fill(HcalDetId const&, int) override;
    void fill(HcalDetId const&, double) override;
    void fill(HcalDetId const&, int, double) override;
    void fill(HcalDetId const&, int, int) override;
    void fill(HcalDetId const&, double, double) override;

    void fill(HcalElectronicsId const&) override;
    void fill(HcalElectronicsId const&, int) override;
    void fill(HcalElectronicsId const&, double) override;
    void fill(HcalElectronicsId const&, int, double) override;
    void fill(HcalElectronicsId const&, int, int) override;
    void fill(HcalElectronicsId const&, double, double) override;

    void fill(HcalTrigTowerDetId const&) override;
    void fill(HcalTrigTowerDetId const&, int) override;
    void fill(HcalTrigTowerDetId const&, double) override;
    void fill(HcalTrigTowerDetId const&, int, int) override;
    void fill(HcalTrigTowerDetId const&, int, double) override;
    void fill(HcalTrigTowerDetId const&, double, double) override;

    void fill(HcalDetId const&, double, double, double);
    void fill(HcalElectronicsId const&, double, double, double);
    void fill(HcalTrigTowerDetId const&, double, double, double);

  protected:
  };
}

#endif
