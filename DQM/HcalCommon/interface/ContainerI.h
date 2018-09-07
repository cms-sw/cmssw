#ifndef ContainerI_h
#define ContainerI_h

#include "DQM/HcalCommon/interface/Container.h"

namespace hcaldqm
{
  class ContainerI : public Container
  {
  public:
    ContainerI():
      Container()
    {}
    ContainerI(std::string const& folder, std::string const& name):
      Container(folder, name)
    {}
    ~ContainerI() override {}

    void initialize(std::string const& folder,
                    std::string const& name, int debug=0) override
    {
      _folder = folder;
      _qname = name;
      _logger.set(_qname, debug);
    }

    virtual void fill(int x)
    {
      _me->Fill(x);
    }

    virtual void book(DQMStore::IBooker& ib,
                      std::string subsystem="Hcal", std::string aux="")
    {
      ib.setCurrentFolder(subsystem+"/"+_folder +aux);
      _me = ib.bookInt(_qname);
    }

  protected:
    MonitorElement* _me;
  };
}

#endif
