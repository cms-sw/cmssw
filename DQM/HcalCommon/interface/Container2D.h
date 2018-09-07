#ifndef Container2D_h
#define Container2D_h

/*
 *      file:           Container2D.h
 *      Author:         Viktor Khristenko
 *
 *      Description:
 *              Container to hold TH2D or like
 *
 */

#include "DQM/HcalCommon/interface/Container1D.h"

#include <vector>
#include <string>

namespace hcaldqm
{
  class Container2D : public Container1D
  {
  public:
    Container2D();
    Container2D(std::string const& folder,
                hashfunctions::HashType, quantity::Quantity*, quantity::Quantity*,
                quantity::Quantity *qz = new quantity::ValueQuantity(quantity::fN));
    ~Container2D() override;

    //	Initialize Container
    //	@folder
    //	@nametitle,
    virtual void initialize(std::string const& folder,
                            hashfunctions::HashType, quantity::Quantity*, quantity::Quantity*,
                            quantity::Quantity *qz = new quantity::ValueQuantity(quantity::fN),
                            int debug=0);

    //	@qname - quantity name replacer
    virtual void initialize(std::string const& folder,
                            std::string const& qname,
                            hashfunctions::HashType, quantity::Quantity*, quantity::Quantity*,
                            quantity::Quantity *qz = new quantity::ValueQuantity(quantity::fN),
                            int debug=0);
    using Container::initialize;
    using Container1D::initialize;
    //	redeclare what to override
    void fill(HcalDetId const&) override ;
    void fill(HcalDetId const&, int) override;
    void fill(HcalDetId const&, double) override;
    void fill(HcalDetId const&, int, double) override;
    void fill(HcalDetId const&, int, int) override;
    void fill(HcalDetId const&, double, double) override;

    double getBinEntries(HcalDetId const&) override;
    double getBinEntries(HcalDetId const&, int) override;
    double getBinEntries(HcalDetId const&, double) override;
    double getBinEntries(HcalDetId const&, int, int) override;
    double getBinEntries(HcalDetId const&, int, double) override;
    double getBinEntries(HcalDetId const&, double, double) override;

    double getBinContent(HcalDetId const&) override;
    double getBinContent(HcalDetId const&, int) override;
    double getBinContent(HcalDetId const&, double) override;
    double getBinContent(HcalDetId const&, int, int) override;
    double getBinContent(HcalDetId const&, int, double) override;
    double getBinContent(HcalDetId const&, double, double) override;

    void setBinContent(HcalDetId const&, int) override;
    void setBinContent(HcalDetId const&, double) override;
    void setBinContent(HcalDetId const&, int, int) override;
    void setBinContent(HcalDetId const&, int, double) override;
    void setBinContent(HcalDetId const&, double, int) override;
    void setBinContent(HcalDetId const&, double, double) override;
    void setBinContent(HcalDetId const&, int, int, int) override;
    void setBinContent(HcalDetId const&, int, double, int) override;
    void setBinContent(HcalDetId const&, double, int, int) override;
    void setBinContent(HcalDetId const&, double, double, int) override;
    void setBinContent(HcalDetId const&, int, int, double) override;
    void setBinContent(HcalDetId const&, int, double, double) override;
    void setBinContent(HcalDetId const&, double, int, double) override;
    void setBinContent(HcalDetId const&, double, double,
                       double) override;

    void fill(HcalElectronicsId const&) override;
    void fill(HcalElectronicsId const&, int) override;
    void fill(HcalElectronicsId const&, double) override;
    void fill(HcalElectronicsId const&, int, double) override;
    void fill(HcalElectronicsId const&, int, int) override;
    void fill(HcalElectronicsId const&, double, double) override;

    double getBinEntries(HcalElectronicsId const&) override;
    double getBinEntries(HcalElectronicsId const&, int) override;
    double getBinEntries(HcalElectronicsId const&, double) override;
    double getBinEntries(HcalElectronicsId const&, int, int) override;
    double getBinEntries(HcalElectronicsId const&, int, double) override;
    double getBinEntries(HcalElectronicsId const&, double,
                         double) override;

    double getBinContent(HcalElectronicsId const&) override;
    double getBinContent(HcalElectronicsId const&, int) override;
    double getBinContent(HcalElectronicsId const&, double) override;
    double getBinContent(HcalElectronicsId const&, int, int) override;
    double getBinContent(HcalElectronicsId const&, int, double) override;
    double getBinContent(HcalElectronicsId const&, double,
                         double) override;

    void setBinContent(HcalElectronicsId const&, int) override;
    void setBinContent(HcalElectronicsId const&, double) override;
    void setBinContent(HcalElectronicsId const&, int, int) override;
    void setBinContent(HcalElectronicsId const&, int, double) override;
    void setBinContent(HcalElectronicsId const&, double, int) override;
    void setBinContent(HcalElectronicsId const&, double, double) override;
    void setBinContent(HcalElectronicsId const&, int, int, int) override;
    void setBinContent(HcalElectronicsId const&, int, double, int) override;
    void setBinContent(HcalElectronicsId const&, double, int, int) override;
    void setBinContent(HcalElectronicsId const&, double, double, int) override;
    void setBinContent(HcalElectronicsId const&, int, int, double) override;
    void setBinContent(HcalElectronicsId const&, int, double, double) override;
    void setBinContent(HcalElectronicsId const&, double, int, double) override;
    void setBinContent(HcalElectronicsId const&, double, double,
                       double) override;

    void fill(HcalTrigTowerDetId const&) override;
    void fill(HcalTrigTowerDetId const&, int) override;
    void fill(HcalTrigTowerDetId const&, double) override;
    void fill(HcalTrigTowerDetId const&, int, int) override;
    void fill(HcalTrigTowerDetId const&, int, double) override;
    void fill(HcalTrigTowerDetId const&, double, double) override;

    double getBinEntries(HcalTrigTowerDetId const&) override;
    double getBinEntries(HcalTrigTowerDetId const&, int) override;
    double getBinEntries(HcalTrigTowerDetId const&, double) override;
    double getBinEntries(HcalTrigTowerDetId const&, int, int) override;
    double getBinEntries(HcalTrigTowerDetId const&, int,
                         double) override;
    double getBinEntries(HcalTrigTowerDetId const&,
                         double, double) override;

    double getBinContent(HcalTrigTowerDetId const&) override;
    double  getBinContent(HcalTrigTowerDetId const&, int) override;
    double getBinContent(HcalTrigTowerDetId const&, double) override;
    double getBinContent(HcalTrigTowerDetId const&, int, int) override;
    double getBinContent(HcalTrigTowerDetId const&, int, double) override;
    double getBinContent(HcalTrigTowerDetId const&,
                         double, double) override;

    void setBinContent(HcalTrigTowerDetId const&, int) override;
    void setBinContent(HcalTrigTowerDetId const&, double) override;
    void setBinContent(HcalTrigTowerDetId const&, int, int) override;
    void setBinContent(HcalTrigTowerDetId const&, int, double) override;
    void setBinContent(HcalTrigTowerDetId const&, double, int) override;
    void setBinContent(HcalTrigTowerDetId const&, double, double) override;
    void setBinContent(HcalTrigTowerDetId const&, int, int, int) override;
    void setBinContent(HcalTrigTowerDetId const&, int, double, int) override;
    void setBinContent(HcalTrigTowerDetId const&, double, int, int) override;
    void setBinContent(HcalTrigTowerDetId const&, double, double, int) override;
    void setBinContent(HcalTrigTowerDetId const&, int, int, double) override;
    void setBinContent(HcalTrigTowerDetId const&, int, double, double) override;
    void setBinContent(HcalTrigTowerDetId const&, double, int, double) override;
    void setBinContent(HcalTrigTowerDetId const&, double, double,
                       double) override;

    //	booking. see Container1D.h
    void book(DQMStore::IBooker&,
              HcalElectronicsMap const*,
              std::string subsystem="Hcal", std::string aux="") override;
    void book(DQMStore::IBooker&,
              HcalElectronicsMap const*, filter::HashFilter const&,
              std::string subsystem="Hcal", std::string aux="") override;

    void showOverflowZ(bool showOverflow);

  protected:
    quantity::Quantity* _qz;

    void customize(MonitorElement*) override;
  };
}


#endif
