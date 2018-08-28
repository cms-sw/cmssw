#ifndef ContainerSingleProf1D_h
#define ContainerSingleProf1D_h

/*
 *      file:                   ContainerSignle1D.h
 *      Author:                 Viktor Khristenko
 *
 *      Description:
 *              Container to hold a single ME - for convenience of initialization
 */

#include "DQM/HcalCommon/interface/ContainerSingle1D.h"

#include <string>

namespace hcaldqm
{
  class ContainerSingleProf1D : public ContainerSingle1D
  {
  public:
    ContainerSingleProf1D();
    ContainerSingleProf1D(std::string const& folder,
                          quantity::Quantity*,
                          quantity::Quantity *qy = new quantity::ValueQuantity(quantity::fN));
    ~ContainerSingleProf1D() override {}

    void initialize(std::string const& folder,
                    quantity::Quantity*,
                    quantity::Quantity *qy = new quantity::ValueQuantity(quantity::fN),
                    int debug=0) override;
    void initialize(std::string const& folder,
                    std::string const&,
                    quantity::Quantity*,
                    quantity::Quantity *qy = new quantity::ValueQuantity(quantity::fN),
                    int debug=0) override;

    //  booking
    void book(DQMStore::IBooker&,
              std::string subsystem="Hcal", std::string aux="") override;
  };
}

#endif
