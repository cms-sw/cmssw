#include "DQM/HcalCommon/interface/ContainerSingleProf2D.h"

namespace hcaldqm
{
  using namespace quantity;
  using namespace constants;
  ContainerSingleProf2D::ContainerSingleProf2D()
  {
    _qx = nullptr;
    _qy = nullptr;
    _qz = nullptr;
  }

  ContainerSingleProf2D::ContainerSingleProf2D(std::string const& folder,
                                               Quantity *qx, Quantity *qy, Quantity *qz):
    ContainerSingle2D(folder, qx, qy, qz)
  {
    _qx->setAxisType(quantity::fXAxis);
    _qy->setAxisType(quantity::fYAxis);
    _qz->setAxisType(quantity::fZAxis);
  }

  /* virtual */ void ContainerSingleProf2D::initialize(std::string const&
                                                       folder, Quantity *qx, Quantity *qy, Quantity *qz,
                                                       int debug/*=0*/)
  {
    ContainerSingle2D::initialize(folder, qx, qy, qz, debug);
    _qx->setAxisType(quantity::fXAxis);
    _qy->setAxisType(quantity::fYAxis);
    _qz->setAxisType(quantity::fZAxis);
  }

  /* virtual */ void ContainerSingleProf2D::initialize(std::string const&
                                                       folder, std::string const& qname,
                                                       Quantity *qx, Quantity *qy, Quantity *qz,
                                                       int debug/*=0*/)
  {
    ContainerSingle2D::initialize(folder, qname, qx, qy, qz, debug);
    _qx->setAxisType(quantity::fXAxis);
    _qy->setAxisType(quantity::fYAxis);
    _qz->setAxisType(quantity::fZAxis);
  }

  /* virtual */ void ContainerSingleProf2D::book(DQMStore::IBooker& ib,
                                                 std::string subsystem, std::string aux)
  {
    ib.setCurrentFolder(subsystem+"/"+_folder+"/"+_qname);
    _me = ib.bookProfile2D(_qname+(aux.empty()?aux:"_"+aux),
                           _qname+(aux.empty()?aux:" "+aux),
                           _qx->nbins(), _qx->min(), _qx->max(),
                           _qy->nbins(), _qy->min(), _qy->max(),
                           _qz->min(), _qz->max());
    customize();
  }

  /* virtual */ void ContainerSingleProf2D::fill(int x, int y)
  {
    _me->Fill(_qx->getValue(x), _qy->getValue(y));
  }

  /* virtual */ void ContainerSingleProf2D::fill(int x, double y)
  {
    _me->Fill(_qx->getValue(x), _qy->getValue(y));
  }

  /* virtual */ void ContainerSingleProf2D::fill(int x, double y, double z)
  {
    _me->Fill(_qx->getValue(x), _qy->getValue(y), _qz->getValue(z));
  }

  /* virtual */ void ContainerSingleProf2D::fill(double x, int y)
  {
    _me->Fill(_qx->getValue(x), _qy->getValue(y));
  }

  /* virtual */ void ContainerSingleProf2D::fill(double x, double y)
  {
    _me->Fill(_qx->getValue(x), _qy->getValue(y));
  }

  /* virtual */ void ContainerSingleProf2D::fill(double x, double y, double z)
  {
    _me->Fill(_qx->getValue(x), _qy->getValue(y), _qz->getValue(z));
  }

  /* virtual */ void ContainerSingleProf2D::fill(int x, int y, double z)
  {
    _me->Fill(_qx->getValue(x), _qy->getValue(y), _qz->getValue(z));
  }

  /* virtual */ void ContainerSingleProf2D::fill(int x, int y, int z)
  {
    _me->Fill(_qx->getValue(x), _qy->getValue(y), _qz->getValue(z));
  }

  /* virtual */ void ContainerSingleProf2D::fill(HcalDetId const& id)
  {
    _me->Fill(_qx->getValue(id), _qy->getValue(id));
  }

  /* virtual */ void ContainerSingleProf2D::fill(HcalDetId const& id, double x)
  {
    if (_qx->isCoordinate() && _qy->isCoordinate())
      _me->Fill(_qx->getValue(id), _qy->getValue(id), x);
    else if (_qx->isCoordinate())
      _me->Fill(_qx->getValue(id), _qy->getValue(x));
    else if (_qy->isCoordinate())
      _me->Fill(_qx->getValue(x), _qy->getValue(id));
  }

  /* virtual */ void ContainerSingleProf2D::fill(HcalDetId const& id, int x)
  {
    if (_qx->isCoordinate() && _qy->isCoordinate())
      _me->Fill(_qx->getValue(id), _qy->getValue(id), x);
    else if (_qx->isCoordinate())
      _me->Fill(_qx->getValue(id), _qy->getValue(x));
    else if (_qy->isCoordinate())
      _me->Fill(_qx->getValue(x), _qy->getValue(id));
  }

  /* virtual */ void ContainerSingleProf2D::fill(HcalDetId const& id, double x,
                                                 double y)
  {
    if (_qx->isCoordinate() && _qy->isCoordinate())
      _me->Fill(_qx->getValue(id), _qy->getValue(id), x);
    else if (_qx->isCoordinate() && !_qy->isCoordinate())
      _me->Fill(_qx->getValue(id), _qy->getValue(x), y);
    else if (!_qx->isCoordinate() && _qy->isCoordinate())
      _me->Fill(_qx->getValue(x), _qy->getValue(id), y);
  }

  /* virtual */ void ContainerSingleProf2D::fill(HcalDetId const& id, int x,
                                                 int y)
  {
    if (_qx->isCoordinate() && _qy->isCoordinate())
      _me->Fill(_qx->getValue(id), _qy->getValue(id), x);
    else if (_qx->isCoordinate() && !_qy->isCoordinate())
      _me->Fill(_qx->getValue(id), _qy->getValue(x), y);
    else if (!_qx->isCoordinate() && _qy->isCoordinate())
      _me->Fill(_qx->getValue(x), _qy->getValue(id), y);
  }

  /* virtual */ void ContainerSingleProf2D::fill(HcalDetId const& id, int x,
                                                 double y)
  {
    if (_qx->isCoordinate() && _qy->isCoordinate())
      _me->Fill(_qx->getValue(id), _qy->getValue(id), x);
    else if (_qx->isCoordinate() && !_qy->isCoordinate())
      _me->Fill(_qx->getValue(id), _qy->getValue(x), y);
    else if (!_qx->isCoordinate() && _qy->isCoordinate())
      _me->Fill(_qx->getValue(x), _qy->getValue(id), y);
  }

  /* virtual */ void ContainerSingleProf2D::fill(HcalElectronicsId const& id)
  {
    _me->Fill(_qx->getValue(id), _qy->getValue(id));
  }

  /* virtual */ void ContainerSingleProf2D::fill(HcalElectronicsId const& id,
                                                 double x)
  {
    if (_qx->isCoordinate() && _qy->isCoordinate())
      _me->Fill(_qx->getValue(id), _qy->getValue(id), x);
    else if (_qx->isCoordinate())
      _me->Fill(_qx->getValue(id), _qy->getValue(x));
    else if (_qy->isCoordinate())
      _me->Fill(_qx->getValue(x), _qy->getValue(id));
  }

  /* virtual */ void ContainerSingleProf2D::fill(HcalElectronicsId const& id,
                                                 int x)
  {
    if (_qx->isCoordinate() && _qy->isCoordinate())
      _me->Fill(_qx->getValue(id), _qy->getValue(id), x);
    else if (_qx->isCoordinate())
      _me->Fill(_qx->getValue(id), _qy->getValue(x));
    else if (_qy->isCoordinate())
      _me->Fill(_qx->getValue(x), _qy->getValue(id));
  }

  /* virtual */ void ContainerSingleProf2D::fill(HcalElectronicsId const& id,
                                                 double x,
                                                 double y)
  {
    if (_qx->isCoordinate() && _qy->isCoordinate())
      _me->Fill(_qx->getValue(id), _qy->getValue(id), x);
    else if (_qx->isCoordinate() && !_qy->isCoordinate())
      _me->Fill(_qx->getValue(id), _qy->getValue(x), y);
    else if (!_qx->isCoordinate() && _qy->isCoordinate())
      _me->Fill(_qx->getValue(x), _qy->getValue(id), y);
  }

  /* virtual */ void ContainerSingleProf2D::fill(HcalElectronicsId const& id,
                                                 int x,
                                                 int y)
  {
    if (_qx->isCoordinate() && _qy->isCoordinate())
      _me->Fill(_qx->getValue(id), _qy->getValue(id), x);
    else if (_qx->isCoordinate() && !_qy->isCoordinate())
      _me->Fill(_qx->getValue(id), _qy->getValue(x), y);
    else if (!_qx->isCoordinate() && _qy->isCoordinate())
      _me->Fill(_qx->getValue(x), _qy->getValue(id), y);
  }

  /* virtual */ void ContainerSingleProf2D::fill(HcalElectronicsId const& id,
                                                 int x,
                                                 double y)
  {
    if (_qx->isCoordinate() && _qy->isCoordinate())
      _me->Fill(_qx->getValue(id), _qy->getValue(id), x);
    else if (_qx->isCoordinate() && !_qy->isCoordinate())
      _me->Fill(_qx->getValue(id), _qy->getValue(x), y);
    else if (!_qx->isCoordinate() && _qy->isCoordinate())
      _me->Fill(_qx->getValue(x), _qy->getValue(id), y);
  }

  /* virtual */ void ContainerSingleProf2D::fill(HcalTrigTowerDetId const& id)
  {
    _me->Fill(_qx->getValue(id), _qy->getValue(id));
  }

  /* virtual */ void ContainerSingleProf2D::fill(HcalTrigTowerDetId const& id,
                                                 double x)
  {
    if (_qx->isCoordinate() && _qy->isCoordinate())
      _me->Fill(_qx->getValue(id), _qy->getValue(id), x);
    else if (_qx->isCoordinate())
      _me->Fill(_qx->getValue(id), _qy->getValue(x));
    else if (_qy->isCoordinate())
      _me->Fill(_qx->getValue(x), _qy->getValue(id));
  }

  /* virtual */ void ContainerSingleProf2D::fill(HcalTrigTowerDetId const& id,
                                                 int x)
  {
    if (_qx->isCoordinate() && _qy->isCoordinate())
      _me->Fill(_qx->getValue(id), _qy->getValue(id), x);
    else if (_qx->isCoordinate())
      _me->Fill(_qx->getValue(id), _qy->getValue(x));
    else if (_qy->isCoordinate())
      _me->Fill(_qx->getValue(x), _qy->getValue(id));
  }

  /* virtual */ void ContainerSingleProf2D::fill(HcalTrigTowerDetId const& id,
                                                 double x, double y)
  {
    if (_qx->isCoordinate() && _qy->isCoordinate())
      _me->Fill(_qx->getValue(id), _qy->getValue(id), x);
    else if (_qx->isCoordinate() && !_qy->isCoordinate())
      _me->Fill(_qx->getValue(id), _qy->getValue(x), y);
    else if (!_qx->isCoordinate() && _qy->isCoordinate())
      _me->Fill(_qx->getValue(x), _qy->getValue(id), y);
  }

  /* virtual */ void ContainerSingleProf2D::fill(HcalTrigTowerDetId const& id,
                                                 int x, int y)
  {
    if (_qx->isCoordinate() && _qy->isCoordinate())
      _me->Fill(_qx->getValue(id), _qy->getValue(id), x);
    else if (_qx->isCoordinate() && !_qy->isCoordinate())
      _me->Fill(_qx->getValue(id), _qy->getValue(x), y);
    else if (!_qx->isCoordinate() && _qy->isCoordinate())
      _me->Fill(_qx->getValue(x), _qy->getValue(id), y);
  }

  /* virtual */ void ContainerSingleProf2D::fill(HcalTrigTowerDetId const& id,
                                                 int x, double y)
  {
    if (_qx->isCoordinate() && _qy->isCoordinate())
      _me->Fill(_qx->getValue(id), _qy->getValue(id), x);
    else if (_qx->isCoordinate() && !_qy->isCoordinate())
      _me->Fill(_qx->getValue(id), _qy->getValue(x), y);
    else if (!_qx->isCoordinate() && _qy->isCoordinate())
      _me->Fill(_qx->getValue(x), _qy->getValue(id), y);
  }

  /* virtual */ void ContainerSingleProf2D::fill(HcalDetId const& did,
                                                 HcalElectronicsId const& eid)
  {
    if (_qx->type()==fDetectorQuantity)
      _me->Fill(_qx->getValue(did), _qy->getValue(eid));
    else
      _me->Fill(_qx->getValue(eid), _qy->getValue(did));
  }

  /* virtual */ void ContainerSingleProf2D::fill(HcalDetId const& did,
                                                 HcalElectronicsId const& eid, double x)
  {
    if (_qx->type()==fDetectorQuantity)
      _me->Fill(_qx->getValue(did), _qy->getValue(eid), x);
    else
      _me->Fill(_qx->getValue(eid), _qy->getValue(did), x);
  }

}
