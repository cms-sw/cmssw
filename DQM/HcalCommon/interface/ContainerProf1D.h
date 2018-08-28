#ifndef ContainerProf1D_h
#define ContainerProf1D_h

/*
 *      file:           ContainerProf1D.h
 *      Author:         Viktor Khristenko
 *
 *      Description:
 *              Container to hold TProfiles.
 *              Direct Inheritance from Container1D + some more funcs
 *
 */

#include "DQM/HcalCommon/interface/Container1D.h"

#include <vector>
#include <string>

namespace hcaldqm
{
  class ContainerProf1D : public Container1D
  {
  public:
    ContainerProf1D();
    ContainerProf1D(std::string const& folder,
                    hashfunctions::HashType,
                    quantity::Quantity *, quantity::Quantity*);
    ~ContainerProf1D() override {}

    void initialize(std::string const& folder,
                    hashfunctions::HashType,
                    quantity::Quantity*, quantity::Quantity*,
                    int debug=0) override;

    void initialize(std::string const& folder,
                    std::string const& qname,
                    hashfunctions::HashType,
                    quantity::Quantity*, quantity::Quantity*,
                    int debug=0) override;

    //  booking
    void book(DQMStore::IBooker&,
              HcalElectronicsMap const*,
              std::string subsystem="Hcal", std::string aux="") override;
    void book(DQMStore::IBooker&,
              HcalElectronicsMap const*, filter::HashFilter const&,
              std::string subsystem="Hcal", std::string aux="") override;
  };
}


#endif
