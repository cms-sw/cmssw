#include "DQM/HcalCommon/interface/ContainerSingleProf1D.h"

namespace hcaldqm
{
  using namespace quantity;
  using namespace constants;
  ContainerSingleProf1D::ContainerSingleProf1D()
  {
    _qx = nullptr;
    _qy = nullptr;
  }

  ContainerSingleProf1D::ContainerSingleProf1D(std::string const& folder,
                                               Quantity *qx, Quantity *qy):
    ContainerSingle1D(folder, qx, qy)
  {
    _qx->setAxisType(quantity::fXAxis);
    _qy->setAxisType(quantity::fYAxis);
  }

  /* virtual */ void ContainerSingleProf1D::initialize(std::string const&
                                                       folder, Quantity *qx, Quantity *qy,
                                                       int debug/*=0*/)
  {
    ContainerSingle1D::initialize(folder, qx, qy, debug);
    _qx->setAxisType(quantity::fXAxis);
    _qy->setAxisType(quantity::fYAxis);
  }

  /* virtual */ void ContainerSingleProf1D::initialize(std::string const&
                                                       folder, std::string const& qname,
                                                       Quantity *qx, Quantity *qy,
                                                       int debug/*=0*/)
  {
    ContainerSingle1D::initialize(folder, qname, qx, qy, debug);
    _qx->setAxisType(quantity::fXAxis);
    _qy->setAxisType(quantity::fYAxis);
  }

  /* virtual */ void ContainerSingleProf1D::book(DQMStore::IBooker& ib,
                                                 std::string subsystem, std::string aux)
  {
    ib.setCurrentFolder(subsystem+"/"+_folder+"/"+_qname);
    _me = ib.bookProfile(_qname+(aux.empty()?aux:"_"+aux),
                         _qname+(aux.empty()?aux:" "+aux),
                         _qx->nbins(), _qx->min(), _qx->max(),
                         _qy->min(), _qy->max());
    customize();
  }
}
