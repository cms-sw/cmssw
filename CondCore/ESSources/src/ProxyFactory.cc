// -*- C++ -*-
//
// Package:     ESSources
// Class  :     ProxyFactory
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Sat Jul 23 19:14:11 EDT 2005
//

// system include files

// user include files
#include "CondCore/ESSources/interface/ProxyFactory.h"
#include "CondCore/ESSources/interface/DataProxy.h"

cond::DataProxyWrapperBase::DataProxyWrapperBase() {}

cond::DataProxyWrapperBase::~DataProxyWrapperBase() {}

void cond::DataProxyWrapperBase::addInfo(std::string const& il, std::string const& cs, std::string const& tag) {
  m_label = il;
  m_connString = cs;
  m_tag = tag;
}

void cond::DataProxyWrapperBase::loadTag(std::string const& tag) {
  m_session.transaction().start(true);
  m_iovProxy = m_session.readIov(tag);
  m_session.transaction().commit();
  m_currentIov.clear();
  m_requests = std::make_shared<std::vector<cond::Iov_t>>();
}

void cond::DataProxyWrapperBase::loadTag(std::string const& tag, boost::posix_time::ptime const& snapshotTime) {
  m_session.transaction().start(true);
  m_iovProxy = m_session.readIov(tag, snapshotTime);
  m_session.transaction().commit();
  m_currentIov.clear();
  m_requests = std::make_shared<std::vector<cond::Iov_t>>();
}

void cond::DataProxyWrapperBase::reload() {
  std::string tag = m_iovProxy.tag();
  if (!tag.empty())
    loadTag(tag);
}

cond::ValidityInterval cond::DataProxyWrapperBase::setIntervalFor(Time_t time) {
  if (!m_currentIov.isValidFor(time)) {
    m_currentIov.clear();
    m_session.transaction().start(true);
    auto it = m_iovProxy.find(time);
    if (it != m_iovProxy.end()) {
      m_currentIov = *it;
    }
    m_session.transaction().commit();
  }
  return cond::ValidityInterval(m_currentIov.since, m_currentIov.till);
}

EDM_REGISTER_PLUGINFACTORY(cond::ProxyFactory, cond::pluginCategory());

namespace cond {
  const char* pluginCategory() { return "CondProxyFactory"; }
}  // namespace cond
