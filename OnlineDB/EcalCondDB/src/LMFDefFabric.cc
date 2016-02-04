#include "OnlineDB/EcalCondDB/interface/LMFDefFabric.h"
#include "OnlineDB/EcalCondDB/interface/LMFColor.h"
#include "OnlineDB/EcalCondDB/interface/LMFTrigType.h"
#include "OnlineDB/EcalCondDB/interface/LMFRunTag.h"
#include "OnlineDB/EcalCondDB/interface/LMFClsVers.h"
#include "OnlineDB/EcalCondDB/interface/LMFPrimVers.h"
#include "OnlineDB/EcalCondDB/interface/LMFSeqVers.h"
#include "OnlineDB/EcalCondDB/interface/LMFCorrVers.h"

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
  throw (std::runtime_error) {
  _lmfColors.clear();
  _lmfTrigTypes.clear();
  _lmfRunTags.clear();
  _lmfPrimVersions.clear();
  _lmfClsVersions.clear();
  _lmfSeqVersions.clear();
  _lmfCorrVersions.clear();
  if ((m_env != NULL) && (m_conn != NULL)) {
    boost::ptr_list<LMFUnique> listOfObjects;
    boost::ptr_list<LMFUnique>::const_iterator i;
    boost::ptr_list<LMFUnique>::const_iterator e;
    listOfObjects = LMFColor(m_env, m_conn).fetchAll();
    i = listOfObjects.begin();
    e = listOfObjects.end();
    while (i != e) {
      LMFColor *c = (LMFColor*)&(*i);
      _lmfColors.push_back(*c);
      i++;
    }
    listOfObjects.clear();
    listOfObjects = LMFTrigType(m_env, m_conn).fetchAll();
    i = listOfObjects.begin();
    e = listOfObjects.end();
    while (i != e) {
      LMFTrigType *c = (LMFTrigType*)&(*i);
      _lmfTrigTypes.push_back(*c);
      i++;
    }
    listOfObjects.clear();
    listOfObjects = LMFRunTag(m_env, m_conn).fetchAll();
    i = listOfObjects.begin();
    e = listOfObjects.end();
    while (i != e) {
      LMFRunTag *c = (LMFRunTag*)&(*i);
      _lmfRunTags.push_back(*c);
      i++;
    }
    listOfObjects.clear();
    listOfObjects = LMFPrimVers(m_env, m_conn).fetchAll();
    i = listOfObjects.begin();
    e = listOfObjects.end();
    while (i != e) {
      LMFPrimVers *c = (LMFPrimVers*)&(*i);
      _lmfPrimVersions.push_back(*c);
      i++;
    }
    listOfObjects.clear();
    listOfObjects = LMFCorrVers(m_env, m_conn).fetchAll();
    i = listOfObjects.begin();
    e = listOfObjects.end();
    while (i != e) {
      LMFCorrVers *c = (LMFCorrVers*)&(*i);
      _lmfCorrVersions.push_back(*c);
      i++;
    }
    listOfObjects.clear();
    listOfObjects = LMFSeqVers(m_env, m_conn).fetchAll();
    i = listOfObjects.begin();
    e = listOfObjects.end();
    while (i != e) {
      LMFSeqVers *c = (LMFSeqVers*)&(*i);
      _lmfSeqVersions.push_back(*c);
      i++;
    }
    listOfObjects.clear();
    listOfObjects = LMFClsVers(m_env, m_conn).fetchAll();
    i = listOfObjects.begin();
    e = listOfObjects.end();
    while (i != e) {
      LMFClsVers *c = (LMFClsVers*)&(*i);
      _lmfClsVersions.push_back(*c);
      i++;
    }
    listOfObjects.clear();
    _lmfColors.sort();
    _lmfTrigTypes.sort();
    _lmfRunTags.sort();
    _lmfPrimVersions.sort();
    _lmfSeqVersions.sort();
    _lmfClsVersions.sort();
    _lmfCorrVersions.sort();
  } else {
    throw(std::runtime_error("LMFDefFabric: cannot initialize since connection not"
			"set"));
  }
}

