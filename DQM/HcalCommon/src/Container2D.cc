#include "DQM/HcalCommon/interface/Container2D.h"
#include "DQM/HcalCommon/interface/Utilities.h"

namespace hcaldqm {
  using namespace constants;
  using namespace quantity;
  using namespace mapper;

  Container2D::Container2D() : _qz(nullptr) {}

  Container2D::Container2D(
      std::string const &folder, hashfunctions::HashType hashtype, Quantity *qx, Quantity *qy, Quantity *qz)
      : Container1D(folder, hashtype, qx, qy), _qz(qz) {
    _qx->setAxisType(quantity::fXAxis);
    _qy->setAxisType(quantity::fYAxis);
    _qz->setAxisType(quantity::fZAxis);
  }

  Container2D::~Container2D() {
    if (_qz != nullptr)
      delete _qz;
    _qz = nullptr;
  }

  /* virtual */ void Container2D::initialize(std::string const &folder,
                                             hashfunctions::HashType hashtype,
                                             Quantity *qx,
                                             Quantity *qy,
                                             Quantity *qz,
                                             int debug /*=0*/) {
    Container1D::initialize(folder, qz->name() + "vs" + qy->name() + "vs" + qx->name(), hashtype, qx, qy, debug);
    _qz = qz;
    _qx->setAxisType(quantity::fXAxis);
    _qy->setAxisType(quantity::fYAxis);
    _qz->setAxisType(quantity::fZAxis);
  }

  /* virtual */ void Container2D::initialize(std::string const &folder,
                                             std::string const &qname,
                                             hashfunctions::HashType hashtype,
                                             Quantity *qx,
                                             Quantity *qy,
                                             Quantity *qz,
                                             int debug /*=0*/) {
    Container1D::initialize(folder, qname, hashtype, qx, qy, debug);
    _qz = qz;
    _qx->setAxisType(quantity::fXAxis);
    _qy->setAxisType(quantity::fYAxis);
    _qz->setAxisType(quantity::fZAxis);
  }

  /* virtual */ void Container2D::fill(HcalDetId const &did) {
    _mes[_hashmap.getHash(did)]->Fill(_qx->getValue(did), _qy->getValue(did));
  }

  //    HcalDetId based
  /* virtual */ void Container2D::fill(HcalDetId const &did, int x) {
    if (_qx->isCoordinate() && _qy->isCoordinate())
      _mes[_hashmap.getHash(did)]->Fill(_qx->getValue(did), _qy->getValue(did), x);
    else if (_qx->isCoordinate())
      _mes[_hashmap.getHash(did)]->Fill(_qx->getValue(did), _qy->getValue(x));
    else if (_qy->isCoordinate())
      _mes[_hashmap.getHash(did)]->Fill(_qx->getValue(x), _qy->getValue(did));
  }

  /* virtual */ void Container2D::fill(HcalDetId const &did, double x) {
    if (_qx->isCoordinate() && _qy->isCoordinate())
      _mes[_hashmap.getHash(did)]->Fill(_qx->getValue(did), _qy->getValue(did), x);
    else if (_qx->isCoordinate())
      _mes[_hashmap.getHash(did)]->Fill(_qx->getValue(did), _qy->getValue(x));
    else if (_qy->isCoordinate())
      _mes[_hashmap.getHash(did)]->Fill(_qx->getValue(x), _qy->getValue(did));
  }

  /* virtual */ void Container2D::fill(HcalDetId const &did, int x, double y) {
    if (_qx->isCoordinate() && !_qy->isCoordinate())
      _mes[_hashmap.getHash(did)]->Fill(_qx->getValue(did), _qy->getValue(x), y);
    else if (!_qx->isCoordinate() && _qy->isCoordinate())
      _mes[_hashmap.getHash(did)]->Fill(_qx->getValue(x), _qy->getValue(did), y);
    else if (!_qx->isCoordinate() && !_qy->isCoordinate())
      _mes[_hashmap.getHash(did)]->Fill(_qx->getValue(x), _qy->getValue(y));
  }

  /* virtual */ void Container2D::fill(HcalDetId const &did, int x, int y) {
    if (_qx->isCoordinate() && !_qy->isCoordinate())
      _mes[_hashmap.getHash(did)]->Fill(_qx->getValue(did), _qy->getValue(x), y);
    else if (!_qx->isCoordinate() && _qy->isCoordinate())
      _mes[_hashmap.getHash(did)]->Fill(_qx->getValue(x), _qy->getValue(did), y);
    else if (!_qx->isCoordinate() && !_qy->isCoordinate())
      _mes[_hashmap.getHash(did)]->Fill(_qx->getValue(x), _qy->getValue(y));
  }

  /* virtual */ void Container2D::fill(HcalDetId const &did, double x, double y) {
    if (_qx->isCoordinate() && !_qy->isCoordinate())
      _mes[_hashmap.getHash(did)]->Fill(_qx->getValue(did), _qy->getValue(x), y);
    else if (!_qx->isCoordinate() && _qy->isCoordinate())
      _mes[_hashmap.getHash(did)]->Fill(_qx->getValue(x), _qy->getValue(did), y);
    else if (!_qx->isCoordinate() && !_qy->isCoordinate())
      _mes[_hashmap.getHash(did)]->Fill(_qx->getValue(x), _qy->getValue(y));
  }

  /* virtual */ double Container2D::getBinEntries(HcalDetId const &id) {
    return _mes[_hashmap.getHash(id)]->getBinEntries(_qx->getBin(id) + _qy->getBin(id) * _qx->wofnbins());
  }

  /* virtual */ double Container2D::getBinEntries(HcalDetId const &id, int x) {
    if (_qx->isCoordinate())
      return _mes[_hashmap.getHash(id)]->getBinEntries(_qx->getBin(id) + _qy->getBin(x) * _qx->wofnbins());
    else
      return _mes[_hashmap.getHash(id)]->getBinEntries(_qx->getBin(x) + _qy->getBin(id) * _qx->wofnbins());
  }

  /* virtual */ double Container2D::getBinEntries(HcalDetId const &id, double x) {
    if (_qx->isCoordinate())
      return _mes[_hashmap.getHash(id)]->getBinEntries(_qx->getBin(id) + _qy->getBin(x) * _qx->wofnbins());
    else
      return _mes[_hashmap.getHash(id)]->getBinEntries(_qx->getBin(x) + _qy->getBin(id) * _qx->wofnbins());
  }

  /* virtual */ double Container2D::getBinEntries(HcalDetId const &id, int x, int y) {
    return _mes[_hashmap.getHash(id)]->getBinEntries(_qx->getBin(x) + _qy->getBin(y) * _qx->wofnbins());
  }

  /* virtual */ double Container2D::getBinEntries(HcalDetId const &id, int x, double y) {
    return _mes[_hashmap.getHash(id)]->getBinEntries(_qx->getBin(x) + _qy->getBin(y) * _qx->wofnbins());
  }

  /* virtual */ double Container2D::getBinEntries(HcalDetId const &id, double x, double y) {
    return _mes[_hashmap.getHash(id)]->getBinEntries(_qx->getBin(x) + _qy->getBin(y) * _qx->wofnbins());
  }

  /* virtual */ double Container2D::getBinContent(HcalDetId const &id) {
    return _mes[_hashmap.getHash(id)]->getBinContent(_qx->getBin(id), _qy->getBin(id));
  }

  /* virtual */ double Container2D::getBinContent(HcalDetId const &id, int x) {
    if (_qx->isCoordinate())
      return _mes[_hashmap.getHash(id)]->getBinContent(_qx->getBin(id), _qy->getBin(x));
    else
      return _mes[_hashmap.getHash(id)]->getBinContent(_qx->getBin(x), _qy->getBin(id));
  }

  /* virtual */ double Container2D::getBinContent(HcalDetId const &id, double x) {
    if (_qx->isCoordinate())
      return _mes[_hashmap.getHash(id)]->getBinContent(_qx->getBin(id), _qy->getBin(x));
    else
      return _mes[_hashmap.getHash(id)]->getBinContent(_qx->getBin(x), _qy->getBin(id));
  }

  /* virtual */ double Container2D::getBinContent(HcalDetId const &id, int x, int y) {
    return _mes[_hashmap.getHash(id)]->getBinContent(_qx->getBin(x), _qy->getBin(y));
  }

  /* virtual */ double Container2D::getBinContent(HcalDetId const &id, int x, double y) {
    return _mes[_hashmap.getHash(id)]->getBinContent(_qx->getBin(x), _qy->getBin(y));
  }

  /* virtual */ double Container2D::getBinContent(HcalDetId const &id, double x, double y) {
    return _mes[_hashmap.getHash(id)]->getBinContent(_qx->getBin(x), _qy->getBin(y));
  }

  //    setBinContent
  /* virtual */ void Container2D::setBinContent(HcalDetId const &id, int x) {
    _mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(id), _qy->getBin(id), x);
  }
  /* virtual */ void Container2D::setBinContent(HcalDetId const &id, double x) {
    _mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(id), _qy->getBin(id), x);
  }
  /* virtual */ void Container2D::setBinContent(HcalDetId const &id, int x, int y) {
    if (_qx->isCoordinate())
      _mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(id), _qy->getBin(x), y);
    else
      _mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(x), _qy->getBin(id), y);
  }

  /* virtual */ void Container2D::setBinContent(HcalDetId const &id, int x, double y) {
    if (_qx->isCoordinate())
      _mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(id), _qy->getBin(x), y);
    else
      _mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(x), _qy->getBin(id), y);
  }
  /* virtual */ void Container2D::setBinContent(HcalDetId const &id, double x, int y) {
    if (_qx->isCoordinate())
      _mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(id), _qy->getBin(x), y);
    else
      _mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(x), _qy->getBin(id), y);
  }
  /* virtual */ void Container2D::setBinContent(HcalDetId const &id, double x, double y) {
    if (_qx->isCoordinate())
      _mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(id), _qy->getBin(x), y);
    else
      _mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(x), _qy->getBin(id), y);
  }

  /* virtual */ void Container2D::setBinContent(HcalDetId const &id, int x, int y, int z) {
    _mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(x), _qy->getBin(y), z);
  }
  /* virtual */ void Container2D::setBinContent(HcalDetId const &id, int x, double y, int z) {
    _mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(x), _qy->getBin(y), z);
  }
  /* virtual */ void Container2D::setBinContent(HcalDetId const &id, double x, int y, int z) {
    _mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(x), _qy->getBin(y), z);
  }
  /* virtual */ void Container2D::setBinContent(HcalDetId const &id, double x, double y, int z) {
    _mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(x), _qy->getBin(y), z);
  }
  /* virtual */ void Container2D::setBinContent(HcalDetId const &id, int x, int y, double z) {
    _mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(x), _qy->getBin(y), z);
  }
  /* virtual */ void Container2D::setBinContent(HcalDetId const &id, int x, double y, double z) {
    _mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(x), _qy->getBin(y), z);
  }
  /* virtual */ void Container2D::setBinContent(HcalDetId const &id, double x, int y, double z) {
    _mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(x), _qy->getBin(y), z);
  }
  /* virtual */ void Container2D::setBinContent(HcalDetId const &id, double x, double y, double z) {
    _mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(x), _qy->getBin(y), z);
  }

  //    by ElectronicsId
  /* virtual */ void Container2D::fill(HcalElectronicsId const &did) {
    _mes[_hashmap.getHash(did)]->Fill(_qx->getValue(did), _qy->getValue(did));
  }

  /* virtual */ void Container2D::fill(HcalElectronicsId const &did, int x) {
    if (_qx->isCoordinate() && _qy->isCoordinate())
      _mes[_hashmap.getHash(did)]->Fill(_qx->getValue(did), _qy->getValue(did), x);
    else if (_qx->isCoordinate())
      _mes[_hashmap.getHash(did)]->Fill(_qx->getValue(did), _qy->getValue(x));
    else if (_qy->isCoordinate())
      _mes[_hashmap.getHash(did)]->Fill(_qx->getValue(x), _qy->getValue(did));
  }

  /* virtual */ void Container2D::fill(HcalElectronicsId const &did, double x) {
    if (_qx->isCoordinate() && _qy->isCoordinate())
      _mes[_hashmap.getHash(did)]->Fill(_qx->getValue(did), _qy->getValue(did), x);
    else if (_qx->isCoordinate())
      _mes[_hashmap.getHash(did)]->Fill(_qx->getValue(did), _qy->getValue(x));
    else if (_qy->isCoordinate())
      _mes[_hashmap.getHash(did)]->Fill(_qx->getValue(x), _qy->getValue(did));
  }

  /* virtual */ void Container2D::fill(HcalElectronicsId const &did, int x, double y) {
    if (_qx->isCoordinate() && !_qy->isCoordinate())
      _mes[_hashmap.getHash(did)]->Fill(_qx->getValue(did), _qy->getValue(x), y);
    else if (!_qx->isCoordinate() && _qy->isCoordinate())
      _mes[_hashmap.getHash(did)]->Fill(_qx->getValue(x), _qy->getValue(did), y);
    else if (!_qx->isCoordinate() && !_qy->isCoordinate())
      _mes[_hashmap.getHash(did)]->Fill(_qx->getValue(x), _qy->getValue(y));
  }

  /* virtual */ void Container2D::fill(HcalElectronicsId const &did, int x, int y) {
    if (_qx->isCoordinate() && !_qy->isCoordinate())
      _mes[_hashmap.getHash(did)]->Fill(_qx->getValue(did), _qy->getValue(x), y);
    else if (!_qx->isCoordinate() && _qy->isCoordinate())
      _mes[_hashmap.getHash(did)]->Fill(_qx->getValue(x), _qy->getValue(did), y);
    else if (!_qx->isCoordinate() && !_qy->isCoordinate())
      _mes[_hashmap.getHash(did)]->Fill(_qx->getValue(x), _qy->getValue(y));
  }

  /* virtual */ void Container2D::fill(HcalElectronicsId const &did, double x, double y) {
    if (_qx->isCoordinate() && !_qy->isCoordinate())
      _mes[_hashmap.getHash(did)]->Fill(_qx->getValue(did), _qy->getValue(x), y);
    else if (!_qx->isCoordinate() && _qy->isCoordinate())
      _mes[_hashmap.getHash(did)]->Fill(_qx->getValue(x), _qy->getValue(did), y);
    else if (!_qx->isCoordinate() && !_qy->isCoordinate())
      _mes[_hashmap.getHash(did)]->Fill(_qx->getValue(x), _qy->getValue(y));
  }

  /* virtual */ double Container2D::getBinEntries(HcalElectronicsId const &id) {
    return _mes[_hashmap.getHash(id)]->getBinEntries(_qx->getBin(id) + _qy->getBin(id) * _qx->wofnbins());
  }

  /* virtual */ double Container2D::getBinEntries(HcalElectronicsId const &id, int x) {
    if (_qx->isCoordinate())
      return _mes[_hashmap.getHash(id)]->getBinEntries(_qx->getBin(id) + _qy->getBin(x) * _qx->wofnbins());
    else
      return _mes[_hashmap.getHash(id)]->getBinEntries(_qx->getBin(x) + _qy->getBin(id) * _qx->wofnbins());
  }

  /* virtual */ double Container2D::getBinEntries(HcalElectronicsId const &id, double x) {
    if (_qx->isCoordinate())
      return _mes[_hashmap.getHash(id)]->getBinEntries(_qx->getBin(id) + _qy->getBin(x) * _qx->wofnbins());
    else
      return _mes[_hashmap.getHash(id)]->getBinEntries(_qx->getBin(x) + _qy->getBin(id) * _qx->wofnbins());
  }

  /* virtual */ double Container2D::getBinEntries(HcalElectronicsId const &id, int x, int y) {
    return _mes[_hashmap.getHash(id)]->getBinEntries(_qx->getBin(x) + _qy->getBin(y) * _qx->wofnbins());
  }

  /* virtual */ double Container2D::getBinEntries(HcalElectronicsId const &id, int x, double y) {
    return _mes[_hashmap.getHash(id)]->getBinEntries(_qx->getBin(x) + _qy->getBin(y) * _qx->wofnbins());
  }

  /* virtual */ double Container2D::getBinEntries(HcalElectronicsId const &id, double x, double y) {
    return _mes[_hashmap.getHash(id)]->getBinEntries(_qx->getBin(x) + _qy->getBin(y) * _qx->wofnbins());
  }

  /* virtual */ double Container2D::getBinContent(HcalElectronicsId const &id) {
    return _mes[_hashmap.getHash(id)]->getBinContent(_qx->getBin(id), _qy->getBin(id));
  }

  /* virtual */ double Container2D::getBinContent(HcalElectronicsId const &id, int x) {
    if (_qx->isCoordinate())
      return _mes[_hashmap.getHash(id)]->getBinContent(_qx->getBin(id), _qy->getBin(x));
    else
      return _mes[_hashmap.getHash(id)]->getBinContent(_qx->getBin(x), _qy->getBin(id));
  }

  /* virtual */ double Container2D::getBinContent(HcalElectronicsId const &id, double x) {
    if (_qx->isCoordinate())
      return _mes[_hashmap.getHash(id)]->getBinContent(_qx->getBin(id), _qy->getBin(x));
    else
      return _mes[_hashmap.getHash(id)]->getBinContent(_qx->getBin(x), _qy->getBin(id));
  }

  /* virtual */ double Container2D::getBinContent(HcalElectronicsId const &id, int x, int y) {
    return _mes[_hashmap.getHash(id)]->getBinContent(_qx->getBin(x), _qy->getBin(y));
  }

  /* virtual */ double Container2D::getBinContent(HcalElectronicsId const &id, int x, double y) {
    return _mes[_hashmap.getHash(id)]->getBinContent(_qx->getBin(x), _qy->getBin(y));
  }

  /* virtual */ double Container2D::getBinContent(HcalElectronicsId const &id, double x, double y) {
    return _mes[_hashmap.getHash(id)]->getBinContent(_qx->getBin(x), _qy->getBin(y));
  }

  //    setBinContent
  /* virtual */ void Container2D::setBinContent(HcalElectronicsId const &id, int x) {
    _mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(id), _qy->getBin(id), x);
  }
  /* virtual */ void Container2D::setBinContent(HcalElectronicsId const &id, double x) {
    _mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(id), _qy->getBin(id), x);
  }
  /* virtual */ void Container2D::setBinContent(HcalElectronicsId const &id, int x, int y) {
    if (_qx->isCoordinate())
      _mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(id), _qy->getBin(x), y);
    else
      _mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(x), _qy->getBin(id), y);
  }

  /* virtual */ void Container2D::setBinContent(HcalElectronicsId const &id, int x, double y) {
    if (_qx->isCoordinate())
      _mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(id), _qy->getBin(x), y);
    else
      _mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(x), _qy->getBin(id), y);
  }
  /* virtual */ void Container2D::setBinContent(HcalElectronicsId const &id, double x, int y) {
    if (_qx->isCoordinate())
      _mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(id), _qy->getBin(x), y);
    else
      _mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(x), _qy->getBin(id), y);
  }
  /* virtual */ void Container2D::setBinContent(HcalElectronicsId const &id, double x, double y) {
    if (_qx->isCoordinate())
      _mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(id), _qy->getBin(x), y);
    else
      _mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(x), _qy->getBin(id), y);
  }

  /* virtual */ void Container2D::setBinContent(HcalElectronicsId const &id, int x, int y, int z) {
    _mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(x), _qy->getBin(y), z);
  }
  /* virtual */ void Container2D::setBinContent(HcalElectronicsId const &id, int x, double y, int z) {
    _mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(x), _qy->getBin(y), z);
  }
  /* virtual */ void Container2D::setBinContent(HcalElectronicsId const &id, double x, int y, int z) {
    _mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(x), _qy->getBin(y), z);
  }
  /* virtual */ void Container2D::setBinContent(HcalElectronicsId const &id, double x, double y, int z) {
    _mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(x), _qy->getBin(y), z);
  }
  /* virtual */ void Container2D::setBinContent(HcalElectronicsId const &id, int x, int y, double z) {
    _mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(x), _qy->getBin(y), z);
  }
  /* virtual */ void Container2D::setBinContent(HcalElectronicsId const &id, int x, double y, double z) {
    _mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(x), _qy->getBin(y), z);
  }
  /* virtual */ void Container2D::setBinContent(HcalElectronicsId const &id, double x, int y, double z) {
    _mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(x), _qy->getBin(y), z);
  }
  /* virtual */ void Container2D::setBinContent(HcalElectronicsId const &id, double x, double y, double z) {
    _mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(x), _qy->getBin(y), z);
  }

  //    by TrigTowerDetId
  /* virtual */ void Container2D::fill(HcalTrigTowerDetId const &did) {
    _mes[_hashmap.getHash(did)]->Fill(_qx->getValue(did), _qy->getValue(did));
  }

  //    HcalDetId based
  /* virtual */ void Container2D::fill(HcalTrigTowerDetId const &did, int x) {
    if (_qx->isCoordinate() && _qy->isCoordinate())
      _mes[_hashmap.getHash(did)]->Fill(_qx->getValue(did), _qy->getValue(did), x);
    else if (_qx->isCoordinate())
      _mes[_hashmap.getHash(did)]->Fill(_qx->getValue(did), _qy->getValue(x));
    else if (_qy->isCoordinate())
      _mes[_hashmap.getHash(did)]->Fill(_qx->getValue(x), _qy->getValue(did));
  }

  /* virtual */ void Container2D::fill(HcalTrigTowerDetId const &did, double x) {
    if (_qx->isCoordinate() && _qy->isCoordinate())
      _mes[_hashmap.getHash(did)]->Fill(_qx->getValue(did), _qy->getValue(did), x);
    else if (_qx->isCoordinate())
      _mes[_hashmap.getHash(did)]->Fill(_qx->getValue(did), _qy->getValue(x));
    else if (_qy->isCoordinate())
      _mes[_hashmap.getHash(did)]->Fill(_qx->getValue(x), _qy->getValue(did));
  }

  /* virtual */ void Container2D::fill(HcalTrigTowerDetId const &did, int x, double y) {
    if (_qx->isCoordinate() && !_qy->isCoordinate())
      _mes[_hashmap.getHash(did)]->Fill(_qx->getValue(did), _qy->getValue(x), y);
    else if (!_qx->isCoordinate() && _qy->isCoordinate())
      _mes[_hashmap.getHash(did)]->Fill(_qx->getValue(x), _qy->getValue(did), y);
    else if (!_qx->isCoordinate() && !_qy->isCoordinate())
      _mes[_hashmap.getHash(did)]->Fill(_qx->getValue(x), _qy->getValue(y));
  }

  /* virtual */ void Container2D::fill(HcalTrigTowerDetId const &did, int x, int y) {
    if (_qx->isCoordinate() && !_qy->isCoordinate())
      _mes[_hashmap.getHash(did)]->Fill(_qx->getValue(did), _qy->getValue(x), y);
    else if (!_qx->isCoordinate() && _qy->isCoordinate())
      _mes[_hashmap.getHash(did)]->Fill(_qx->getValue(x), _qy->getValue(did), y);
    else if (!_qx->isCoordinate() && !_qy->isCoordinate())
      _mes[_hashmap.getHash(did)]->Fill(_qx->getValue(x), _qy->getValue(y));
  }

  /* virtual */ void Container2D::fill(HcalTrigTowerDetId const &did, double x, double y) {
    if (_qx->isCoordinate() && !_qy->isCoordinate())
      _mes[_hashmap.getHash(did)]->Fill(_qx->getValue(did), _qy->getValue(x), y);
    else if (!_qx->isCoordinate() && _qy->isCoordinate())
      _mes[_hashmap.getHash(did)]->Fill(_qx->getValue(x), _qy->getValue(did), y);
    else if (!_qx->isCoordinate() && !_qy->isCoordinate())
      _mes[_hashmap.getHash(did)]->Fill(_qx->getValue(x), _qy->getValue(y));
  }

  void Container2D::fill(HcalTrigTowerDetId const &did, HcalElectronicsId const &eid, int x, int y) {
    if (_qx->isCoordinate() && !_qy->isCoordinate())
      _mes[_hashmap.getHash(did, eid)]->Fill(_qx->getValue(did), _qy->getValue(x), y);
    else if (!_qx->isCoordinate() && _qy->isCoordinate())
      _mes[_hashmap.getHash(did, eid)]->Fill(_qx->getValue(x), _qy->getValue(did), y);
    else if (!_qx->isCoordinate() && !_qy->isCoordinate())
      _mes[_hashmap.getHash(did, eid)]->Fill(_qx->getValue(x), _qy->getValue(y));
  }

  /* virtual */ double Container2D::getBinEntries(HcalTrigTowerDetId const &id) {
    return _mes[_hashmap.getHash(id)]->getBinEntries(_qx->getBin(id) + _qy->getBin(id) * _qx->wofnbins());
  }

  /* virtual */ double Container2D::getBinEntries(HcalTrigTowerDetId const &id, int x) {
    if (_qx->isCoordinate())
      return _mes[_hashmap.getHash(id)]->getBinEntries(_qx->getBin(id) + _qy->getBin(x) * _qx->wofnbins());
    else
      return _mes[_hashmap.getHash(id)]->getBinEntries(_qx->getBin(x) + _qy->getBin(id) * _qx->wofnbins());
  }

  /* virtual */ double Container2D::getBinEntries(HcalTrigTowerDetId const &id, double x) {
    if (_qx->isCoordinate())
      return _mes[_hashmap.getHash(id)]->getBinEntries(_qx->getBin(id) + _qy->getBin(x) * _qx->wofnbins());
    else
      return _mes[_hashmap.getHash(id)]->getBinEntries(_qx->getBin(x) + _qy->getBin(id) * _qx->wofnbins());
  }

  /* virtual */ double Container2D::getBinEntries(HcalTrigTowerDetId const &id, int x, int y) {
    return _mes[_hashmap.getHash(id)]->getBinEntries(_qx->getBin(x) + _qy->getBin(y) * _qx->wofnbins());
  }

  /* virtual */ double Container2D::getBinEntries(HcalTrigTowerDetId const &id, int x, double y) {
    return _mes[_hashmap.getHash(id)]->getBinEntries(_qx->getBin(x) + _qy->getBin(y) * _qx->wofnbins());
  }

  /* virtual */ double Container2D::getBinEntries(HcalTrigTowerDetId const &id, double x, double y) {
    return _mes[_hashmap.getHash(id)]->getBinEntries(_qx->getBin(x) + _qy->getBin(y) * _qx->wofnbins());
  }

  /* virtual */ double Container2D::getBinContent(HcalTrigTowerDetId const &id) {
    return _mes[_hashmap.getHash(id)]->getBinContent(_qx->getBin(id), _qy->getBin(id));
  }

  /* virtual */ double Container2D::getBinContent(HcalTrigTowerDetId const &id, int x) {
    if (_qx->isCoordinate())
      return _mes[_hashmap.getHash(id)]->getBinContent(_qx->getBin(id), _qy->getBin(x));
    else
      return _mes[_hashmap.getHash(id)]->getBinContent(_qx->getBin(x), _qy->getBin(id));
  }

  /* virtual */ double Container2D::getBinContent(HcalTrigTowerDetId const &id, double x) {
    if (_qx->isCoordinate())
      return _mes[_hashmap.getHash(id)]->getBinContent(_qx->getBin(id), _qy->getBin(x));
    else
      return _mes[_hashmap.getHash(id)]->getBinContent(_qx->getBin(x), _qy->getBin(id));
  }

  /* virtual */ double Container2D::getBinContent(HcalTrigTowerDetId const &id, int x, int y) {
    return _mes[_hashmap.getHash(id)]->getBinContent(_qx->getBin(x), _qy->getBin(y));
  }

  /* virtual */ double Container2D::getBinContent(HcalTrigTowerDetId const &id, int x, double y) {
    return _mes[_hashmap.getHash(id)]->getBinContent(_qx->getBin(x), _qy->getBin(y));
  }

  /* virtual */ double Container2D::getBinContent(HcalTrigTowerDetId const &id, double x, double y) {
    return _mes[_hashmap.getHash(id)]->getBinContent(_qx->getBin(x), _qy->getBin(y));
  }

  //    setBinContent
  /* virtual */ void Container2D::setBinContent(HcalTrigTowerDetId const &id, int x) {
    _mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(id), _qy->getBin(id), x);
  }
  /* virtual */ void Container2D::setBinContent(HcalTrigTowerDetId const &id, double x) {
    _mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(id), _qy->getBin(id), x);
  }
  /* virtual */ void Container2D::setBinContent(HcalTrigTowerDetId const &id, int x, int y) {
    if (_qx->isCoordinate())
      _mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(id), _qy->getBin(x), y);
    else
      _mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(x), _qy->getBin(id), y);
  }

  /* virtual */ void Container2D::setBinContent(HcalTrigTowerDetId const &id, int x, double y) {
    if (_qx->isCoordinate())
      _mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(id), _qy->getBin(x), y);
    else
      _mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(x), _qy->getBin(id), y);
  }
  /* virtual */ void Container2D::setBinContent(HcalTrigTowerDetId const &id, double x, int y) {
    if (_qx->isCoordinate())
      _mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(id), _qy->getBin(x), y);
    else
      _mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(x), _qy->getBin(id), y);
  }
  /* virtual */ void Container2D::setBinContent(HcalTrigTowerDetId const &id, double x, double y) {
    if (_qx->isCoordinate())
      _mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(id), _qy->getBin(x), y);
    else
      _mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(x), _qy->getBin(id), y);
  }

  /* virtual */ void Container2D::setBinContent(HcalTrigTowerDetId const &id, int x, int y, int z) {
    _mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(x), _qy->getBin(y), z);
  }
  /* virtual */ void Container2D::setBinContent(HcalTrigTowerDetId const &id, int x, double y, int z) {
    _mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(x), _qy->getBin(y), z);
  }
  /* virtual */ void Container2D::setBinContent(HcalTrigTowerDetId const &id, double x, int y, int z) {
    _mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(x), _qy->getBin(y), z);
  }
  /* virtual */ void Container2D::setBinContent(HcalTrigTowerDetId const &id, double x, double y, int z) {
    _mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(x), _qy->getBin(y), z);
  }
  /* virtual */ void Container2D::setBinContent(HcalTrigTowerDetId const &id, int x, int y, double z) {
    _mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(x), _qy->getBin(y), z);
  }
  /* virtual */ void Container2D::setBinContent(HcalTrigTowerDetId const &id, int x, double y, double z) {
    _mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(x), _qy->getBin(y), z);
  }
  /* virtual */ void Container2D::setBinContent(HcalTrigTowerDetId const &id, double x, int y, double z) {
    _mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(x), _qy->getBin(y), z);
  }
  /* virtual */ void Container2D::setBinContent(HcalTrigTowerDetId const &id, double x, double y, double z) {
    _mes[_hashmap.getHash(id)]->setBinContent(_qx->getBin(x), _qy->getBin(y), z);
  }

  //    booking
  /* virtual */ void Container2D::book(DQMStore::IBooker &ib,
                                       HcalElectronicsMap const *emap,
                                       std::string subsystem,
                                       std::string aux) {
    //  full path as in Container1D.cc
    //
    ib.setCurrentFolder(subsystem + "/" + _folder + "/" + _qname + (aux.empty() ? aux : "_" + aux) + "/" +
                        _hashmap.getHashTypeName());
    if (_hashmap.isDHash()) {
      //      for Detector Hashes
      std::vector<HcalGenericDetId> dids = emap->allPrecisionId();
      for (std::vector<HcalGenericDetId>::const_iterator it = dids.begin(); it != dids.end(); ++it) {
        //  skip trigger towers and calib
        if (!it->isHcalDetId())
          continue;

        HcalDetId did = HcalDetId(it->rawId());
        uint32_t hash = _hashmap.getHash(did);
        MEMap::iterator mit = _mes.find(hash);
        if (mit != _mes.end())
          continue;

        _logger.debug(_hashmap.getName(did));
        _mes.insert(std::make_pair(hash,
                                   ib.book2D(_hashmap.getName(did),
                                             _hashmap.getName(did),
                                             _qx->nbins(),
                                             _qx->min(),
                                             _qx->max(),
                                             _qy->nbins(),
                                             _qy->min(),
                                             _qy->max())));
        customize(_mes[hash]);
      }
    } else if (_hashmap.isEHash()) {
      //      for Electronics hashes
      std::vector<HcalElectronicsId> eids = emap->allElectronicsIdPrecision();
      for (std::vector<HcalElectronicsId>::const_iterator it = eids.begin(); it != eids.end(); ++it) {
        HcalElectronicsId eid = HcalElectronicsId(it->rawId());
        uint32_t hash = _hashmap.getHash(eid);
        MEMap::iterator mit = _mes.find(hash);
        if (mit != _mes.end())
          continue;

        _logger.debug(_hashmap.getName(eid));
        _mes.insert(std::make_pair(hash,
                                   ib.book2D(_hashmap.getName(eid),
                                             _hashmap.getName(eid),
                                             _qx->nbins(),
                                             _qx->min(),
                                             _qx->max(),
                                             _qy->nbins(),
                                             _qy->min(),
                                             _qy->max())));
        customize(_mes[hash]);
      }
    } else if (_hashmap.isTHash()) {
      //      for TrigTower hashes
      std::vector<HcalTrigTowerDetId> tids = emap->allTriggerId();
      for (std::vector<HcalTrigTowerDetId>::const_iterator it = tids.begin(); it != tids.end(); ++it) {
        HcalTrigTowerDetId tid = HcalTrigTowerDetId(it->rawId());
        _logger.debug(_hashmap.getName(tid));
        uint32_t hash = _hashmap.getHash(tid);
        MEMap::iterator mit = _mes.find(hash);
        if (mit != _mes.end())
          continue;

        _logger.debug(_hashmap.getName(tid));
        _mes.insert(std::make_pair(hash,
                                   ib.book2D(_hashmap.getName(tid),
                                             _hashmap.getName(tid),
                                             _qx->nbins(),
                                             _qx->min(),
                                             _qx->max(),
                                             _qy->nbins(),
                                             _qy->min(),
                                             _qy->max())));

        customize(_mes[hash]);
      }
    } else if (_hashmap.isMixHash()) {
      //      for Mixed hashes
      std::vector<HcalTrigTowerDetId> tids = emap->allTriggerId();
      for (std::vector<HcalTrigTowerDetId>::const_iterator it = tids.begin(); it != tids.end(); ++it) {
        HcalTrigTowerDetId tid = HcalTrigTowerDetId(it->rawId());
        HcalElectronicsId eid = HcalElectronicsId(emap->lookupTrigger(*it).rawId());
        _logger.debug(_hashmap.getName(tid, eid));
        uint32_t hash = _hashmap.getHash(tid, eid);
        MEMap::iterator mit = _mes.find(hash);
        if (mit != _mes.end())
          continue;

        _logger.debug(_hashmap.getName(tid, eid));
        _mes.insert(std::make_pair(hash,
                                   ib.book2D(_hashmap.getName(tid, eid),
                                             _hashmap.getName(tid, eid),
                                             _qx->nbins(),
                                             _qx->min(),
                                             _qx->max(),
                                             _qy->nbins(),
                                             _qy->min(),
                                             _qy->max())));

        customize(_mes[hash]);
      }
    }
  }

  /* virtual */ void Container2D::book(DQMStore::IBooker &ib,
                                       HcalElectronicsMap const *emap,
                                       filter::HashFilter const &filter,
                                       std::string subsystem,
                                       std::string aux) {
    //  full path as in Container1D.cc
    //
    ib.setCurrentFolder(subsystem + "/" + _folder + "/" + _qname + (aux.empty() ? aux : "_" + aux) + "/" +
                        _hashmap.getHashTypeName());
    if (_hashmap.isDHash()) {
      //      for Detector Hashes
      std::vector<HcalGenericDetId> dids = emap->allPrecisionId();
      for (std::vector<HcalGenericDetId>::const_iterator it = dids.begin(); it != dids.end(); ++it) {
        //  skip trigger towers and calib
        if (!it->isHcalDetId())
          continue;

        HcalDetId did = HcalDetId(it->rawId());
        uint32_t hash = _hashmap.getHash(did);
        MEMap::iterator mit = _mes.find(hash);
        if (mit != _mes.end())
          continue;
        if (filter.filter(did))
          continue;

        _logger.debug(_hashmap.getName(did));
        _mes.insert(std::make_pair(hash,
                                   ib.book2D(_hashmap.getName(did),
                                             _hashmap.getName(did),
                                             _qx->nbins(),
                                             _qx->min(),
                                             _qx->max(),
                                             _qy->nbins(),
                                             _qy->min(),
                                             _qy->max())));
        customize(_mes[hash]);
      }
    } else if (_hashmap.isEHash()) {
      //      for Electronics hashes
      std::vector<HcalElectronicsId> eids = emap->allElectronicsIdPrecision();
      for (std::vector<HcalElectronicsId>::const_iterator it = eids.begin(); it != eids.end(); ++it) {
        HcalElectronicsId eid = HcalElectronicsId(it->rawId());
        uint32_t hash = _hashmap.getHash(eid);
        MEMap::iterator mit = _mes.find(hash);
        if (mit != _mes.end())
          continue;
        if (filter.filter(eid))
          continue;

        _logger.debug(_hashmap.getName(eid));
        _mes.insert(std::make_pair(hash,
                                   ib.book2D(_hashmap.getName(eid),
                                             _hashmap.getName(eid),
                                             _qx->nbins(),
                                             _qx->min(),
                                             _qx->max(),
                                             _qy->nbins(),
                                             _qy->min(),
                                             _qy->max())));
        customize(_mes[hash]);
      }
    } else if (_hashmap.isTHash()) {
      //      for TrigTower hashes
      std::vector<HcalTrigTowerDetId> tids = emap->allTriggerId();
      for (std::vector<HcalTrigTowerDetId>::const_iterator it = tids.begin(); it != tids.end(); ++it) {
        HcalTrigTowerDetId tid = HcalTrigTowerDetId(it->rawId());
        _logger.debug(_hashmap.getName(tid));
        uint32_t hash = _hashmap.getHash(tid);
        MEMap::iterator mit = _mes.find(hash);
        if (mit != _mes.end())
          continue;
        if (filter.filter(tid))
          continue;

        _logger.debug(_hashmap.getName(tid));
        _mes.insert(std::make_pair(hash,
                                   ib.book2D(_hashmap.getName(tid),
                                             _hashmap.getName(tid),
                                             _qx->nbins(),
                                             _qx->min(),
                                             _qx->max(),
                                             _qy->nbins(),
                                             _qy->min(),
                                             _qy->max())));

        customize(_mes[hash]);
      }
    } else if (_hashmap.isMixHash()) {
      //      for Mixed hashes
      std::vector<HcalTrigTowerDetId> tids = emap->allTriggerId();
      for (std::vector<HcalTrigTowerDetId>::const_iterator it = tids.begin(); it != tids.end(); ++it) {
        HcalTrigTowerDetId tid = HcalTrigTowerDetId(it->rawId());
        HcalElectronicsId eid = HcalElectronicsId(emap->lookupTrigger(*it).rawId());
        _logger.debug(_hashmap.getName(tid, eid));
        uint32_t hash = _hashmap.getHash(tid, eid);
        MEMap::iterator mit = _mes.find(hash);
        if (mit != _mes.end())
          continue;
        if (filter.filter(tid))
          continue;

        _logger.debug(_hashmap.getName(tid, eid));
        _mes.insert(std::make_pair(hash,
                                   ib.book2D(_hashmap.getName(tid, eid),
                                             _hashmap.getName(tid, eid),
                                             _qx->nbins(),
                                             _qx->min(),
                                             _qx->max(),
                                             _qy->nbins(),
                                             _qy->min(),
                                             _qy->max())));

        customize(_mes[hash]);
      }
    }
  }

  /* virtual */ void Container2D::customize(MonitorElement *me) {
    me->setAxisTitle(_qx->name(), 1);
    me->setAxisTitle(_qy->name(), 2);
    me->setAxisTitle(_qz->name(), 3);

    TH1 *h = me->getTH1();
    _qx->setBits(h);
    _qy->setBits(h);
    _qz->setBits(h);

    std::vector<std::string> xlabels = _qx->getLabels();
    std::vector<std::string> ylabels = _qy->getLabels();
    for (unsigned int i = 0; i < xlabels.size(); i++) {
      me->setBinLabel(i + 1, xlabels[i], 1);
    }
    for (unsigned int i = 0; i < ylabels.size(); i++) {
      me->setBinLabel(i + 1, ylabels[i], 2);
    }
  }

  void Container2D::showOverflowZ(bool showOverflow) { _qz->showOverflow(showOverflow); }
}  // namespace hcaldqm
