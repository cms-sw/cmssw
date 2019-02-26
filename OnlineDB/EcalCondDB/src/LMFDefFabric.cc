#include "OnlineDB/EcalCondDB/interface/LMFDefFabric.h"
#include "OnlineDB/EcalCondDB/interface/LMFColor.h"
#include "OnlineDB/EcalCondDB/interface/LMFTrigType.h"
#include "OnlineDB/EcalCondDB/interface/LMFRunTag.h"
#include "OnlineDB/EcalCondDB/interface/LMFPrimVers.h"
#include "OnlineDB/EcalCondDB/interface/LMFSeqVers.h"

#include <iostream>

LMFDefFabric::LMFDefFabric() {
  noDebug();
}

LMFDefFabric::LMFDefFabric(oracle::occi::Environment* env,
                           oracle::occi::Connection* conn) {
  noDebug();
  setConnection(env, conn);
  initialize();
}

LMFDefFabric::LMFDefFabric(EcalDBConnection *c) {
  noDebug();
  setConnection(c->getEnv(), c->getConn());
  initialize();
}

LMFDefFabric::~LMFDefFabric() {
}

void LMFDefFabric::debug() {
  _debug = true;
}

void LMFDefFabric::noDebug() {
  _debug = false;
}

std::list<LMFColor> LMFDefFabric::getColors() const {
  return _lmfColors;
}

std::list<LMFTrigType> LMFDefFabric::getTriggerTypes() const {
  return _lmfTrigTypes;
}

std::list<LMFRunTag> LMFDefFabric::getRunTags() const {
  return _lmfRunTags;
}

LMFColor LMFDefFabric::getColor(std::string name) const {
  std::list<LMFColor>::const_iterator i = _lmfColors.begin();
  std::list<LMFColor>::const_iterator e = _lmfColors.end();
  LMFColor ret;
  bool loop = true;
  while ((loop) && (i != e)) {
    if (i->getShortName() == name) {
      ret = *i;
      loop = false;
    }
    i++;
  }
  return ret;
}

LMFColor LMFDefFabric::getColorFromID(int id) const {
  std::list<LMFColor>::const_iterator i = _lmfColors.begin();
  std::list<LMFColor>::const_iterator e = _lmfColors.end();
  LMFColor ret;
  bool loop = true;
  while ((loop) && (i != e)) {
    if (i->getID() == id) {
      ret = *i;
      loop = false;
    }
    i++;
  }
  return ret;
}

LMFColor LMFDefFabric::getColor(int index) const {
  std::list<LMFColor>::const_iterator i = _lmfColors.begin();
  std::list<LMFColor>::const_iterator e = _lmfColors.end();
  LMFColor ret;
  bool loop = true;
  while ((loop) && (i != e)) {
    if (i->getColorIndex() == index) {
      ret = *i;
      loop = false;
    }
    i++;
  }
  return ret;
}

int LMFDefFabric::getColorID(std::string sname) const {
  return getColor(sname).getID();
}

int LMFDefFabric::getColorID(int index) const {
  return getColor(index).getID();
}

int LMFDefFabric::getTrigTypeID(std::string sname) const {
  return getTrigType(sname).getID();
}

LMFTrigType LMFDefFabric::getTrigType(std::string sname) const {
  std::list<LMFTrigType>::const_iterator i = _lmfTrigTypes.begin();
  std::list<LMFTrigType>::const_iterator e = _lmfTrigTypes.end();
  LMFTrigType tt;
  bool loop = true;
  while ((loop) && (i != e)) {
    if (i->getShortName() == sname) {
      tt = *i;
      loop = false;
    }
    i++;
  }
  return tt;
}

LMFTrigType LMFDefFabric::getTrigTypeFromID(int id) const {
  std::list<LMFTrigType>::const_iterator i = _lmfTrigTypes.begin();
  std::list<LMFTrigType>::const_iterator e = _lmfTrigTypes.end();
  LMFTrigType tt;
  bool loop = true;
  while ((loop) && (i != e)) {
    if (i->getID() == id) {
      tt = *i;
      loop = false;
    }
    i++;
  }
  return tt;
}

LMFRunTag LMFDefFabric::getRunTag(std::string tag, int version) const {
  std::list<LMFRunTag>::const_iterator i = _lmfRunTags.begin();
  std::list<LMFRunTag>::const_iterator e = _lmfRunTags.end();
  LMFRunTag rt;
  bool loop = true;
  while ((loop) && (i != e)) {
    if ((i->getGeneralTag()) == tag && (i->getVersion() == version)) {
      rt = *i;
    }
    i++;
  }
  return rt;
}

LMFRunTag LMFDefFabric::getRunTagFromID(int id) const {
  std::list<LMFRunTag>::const_iterator i = _lmfRunTags.begin();
  std::list<LMFRunTag>::const_iterator e = _lmfRunTags.end();
  LMFRunTag rt;
  bool loop = true;
  while ((loop) && (i != e)) {
    if (i->getID() == id) {
      rt = *i;
    }
    i++;
  }
  return rt;
}

int LMFDefFabric::getRunTagID(std::string tag, int version) const {
  return getRunTag(tag, version).getID();
}

void LMFDefFabric::initialize() 
  noexcept(false) {
  _lmfColors.clear();
  _lmfTrigTypes.clear();
  _lmfRunTags.clear();
  _lmfPrimVersions.clear();
  _lmfSeqVersions.clear();
  _lmfCorrVersions.clear();
  if ((m_env != nullptr) && (m_conn != nullptr)) {
    boost::ptr_list<LMFUnique> listOfObjects;
    boost::ptr_list<LMFUnique>::const_iterator i;
    boost::ptr_list<LMFUnique>::const_iterator e;
    listOfObjects = LMFColor(m_env, m_conn).fetchAll();
    i = listOfObjects.begin();
    e = listOfObjects.end();
    while (i != e) {
      const LMFColor *c = static_cast<const LMFColor*>(&(*i));
      _lmfColors.push_back(*c);
      i++;
    }
    listOfObjects.clear();
    listOfObjects = LMFTrigType(m_env, m_conn).fetchAll();
    i = listOfObjects.begin();
    e = listOfObjects.end();
    while (i != e) {
      const LMFTrigType *c = static_cast<const LMFTrigType*>(&(*i));
      _lmfTrigTypes.push_back(*c);
      i++;
    }
    listOfObjects.clear();
    listOfObjects = LMFRunTag(m_env, m_conn).fetchAll();
    i = listOfObjects.begin();
    e = listOfObjects.end();
    while (i != e) {
      const LMFRunTag *c = static_cast<const LMFRunTag*>(&(*i));
      _lmfRunTags.push_back(*c);
      i++;
    }
    listOfObjects.clear();
    listOfObjects = LMFPrimVers(m_env, m_conn).fetchAll();
    i = listOfObjects.begin();
    e = listOfObjects.end();
    while (i != e) {
      const LMFPrimVers *c = static_cast<const LMFPrimVers*>(&(*i));
      _lmfPrimVersions.push_back(*c);
      i++;
    }
    listOfObjects.clear();
    listOfObjects = LMFCorrVers(m_env, m_conn).fetchAll();
    i = listOfObjects.begin();
    e = listOfObjects.end();
    while (i != e) {
      const LMFCorrVers *c = static_cast<const LMFCorrVers*>(&(*i));
      _lmfCorrVersions.push_back(*c);
      i++;
    }
    listOfObjects.clear();
    listOfObjects = LMFSeqVers(m_env, m_conn).fetchAll();
    i = listOfObjects.begin();
    e = listOfObjects.end();
    while (i != e) {
      const LMFSeqVers *c = static_cast<const LMFSeqVers*>(&(*i));
      _lmfSeqVersions.push_back(*c);
      i++;
    }
    listOfObjects.clear();
    _lmfColors.sort();
    _lmfTrigTypes.sort();
    _lmfRunTags.sort();
    _lmfPrimVersions.sort();
    _lmfSeqVersions.sort();
    _lmfCorrVersions.sort();
  } else {
    throw(std::runtime_error("LMFDefFabric: cannot initialize since connection not"
			"set"));
  }
}

void LMFDefFabric::dump() {
  std::cout << "========= Fabric dump @ address " << this << " ============"
	    << std::endl;
  std::list<LMFColor>::const_iterator    i1 = _lmfColors.begin();
  std::list<LMFTrigType>::const_iterator i2 = _lmfTrigTypes.begin();
  std::list<LMFRunTag>::const_iterator   i3 = _lmfRunTags.begin();
  std::list<LMFPrimVers>::const_iterator i4 = _lmfPrimVersions.begin();
  std::list<LMFSeqVers>::const_iterator  i5 = _lmfSeqVersions.begin();
  std::list<LMFCorrVers>::const_iterator i6 = _lmfCorrVersions.begin();
  std::list<LMFColor>::const_iterator    e1 = _lmfColors.end();
  std::list<LMFTrigType>::const_iterator e2 = _lmfTrigTypes.end();
  std::list<LMFRunTag>::const_iterator   e3 = _lmfRunTags.end();
  std::list<LMFPrimVers>::const_iterator e4 = _lmfPrimVersions.end();
  std::list<LMFSeqVers>::const_iterator  e5 = _lmfSeqVersions.end();
  std::list<LMFCorrVers>::const_iterator e6 = _lmfCorrVersions.end();
  std::cout << "=== Colors" << std::endl;
  while (i1 != e1) {
    i1++->dump();
  }
  std::cout << "=== Trigger Types" << std::endl;
  while (i2 != e2) {
    i2++->dump();
  }
  std::cout << "=== Run Tags" << std::endl;
  while (i3 != e3) {
    i3++->dump();
  }
  std::cout << "=== Prim. Vers." << std::endl;
  while (i4 != e4) {
    i4++->dump();
  }
  std::cout << "=== Seq. Vers." << std::endl;
  while (i5 != e5) {
    i5++->dump();
  }
  std::cout << "=== Corr. Vers." << std::endl;
  while (i6 != e6) {
    i6++->dump();
  }
}
