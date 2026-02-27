// -*- C++ -*-
//
// Package:     ESSources
// Class  :     ProductResolverFactory
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Sat Jul 23 19:14:11 EDT 2005
//

// system include files

// user include files
#include "CondCore/ESSources/interface/ProductResolverFactory.h"
#include "CondCore/ESSources/interface/ProductResolver.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

cond::ProductResolverWrapperBase::ProductResolverWrapperBase() {}

cond::ProductResolverWrapperBase::~ProductResolverWrapperBase() {}

void cond::ProductResolverWrapperBase::addInfo(std::string const& il, std::string const& cs, std::string const& tag) {
  m_label = il;
  m_connString = cs;
  m_tag = tag;
}

void cond::ProductResolverWrapperBase::loadTag(std::string const& tag) {
  m_session.transaction().start(true);
  m_iovProxy = m_session.readIov(tag);
  m_iovProxy.setPrintDebug(m_printDebug);
  m_session.transaction().commit();
  m_currentIov.clear();
  m_requests = std::make_shared<std::vector<cond::Iov_t>>();
  if (m_printDebug) {
    edm::LogSystem("ProductResolverWrapperBase") << "loadTag executed with tag: " << tag;
  }
}

void cond::ProductResolverWrapperBase::loadTag(std::string const& tag, boost::posix_time::ptime const& snapshotTime) {
  m_session.transaction().start(true);
  m_iovProxy = m_session.readIov(tag, snapshotTime);
  m_iovProxy.setPrintDebug(m_printDebug);
  m_session.transaction().commit();
  m_currentIov.clear();
  m_requests = std::make_shared<std::vector<cond::Iov_t>>();
  if (m_printDebug) {
    edm::LogSystem("ProductResolverWrapperBase")
        << "loadTag executed with tag: " << tag << " and snapshotTime: " << snapshotTime;
  }
}

void cond::ProductResolverWrapperBase::reload() {
  std::string tag = m_iovProxy.tagInfo().name;
  if (!tag.empty())
    loadTag(tag);
}

cond::ValidityInterval cond::ProductResolverWrapperBase::setIntervalFor(Time_t time) {
  if (!m_currentIov.isValidFor(time)) {
    m_currentIov.clear();
    m_session.transaction().start(true);
    m_currentIov = m_iovProxy.getInterval(time);
    m_session.transaction().commit();
  }
  if (m_printDebug) {
    edm::LogSystem("ProductResolverWrapperBase")
        << "setIntervalFor for tag:" << m_iovProxy.tagInfo().name << " executed with time: " << time << "\n"
        << " set ValidityInterval: since: " << m_currentIov.since << " till: " << m_currentIov.till;
  }
  return cond::ValidityInterval(m_currentIov.since, m_currentIov.till);
}

EDM_REGISTER_PLUGINFACTORY(cond::ProductResolverFactory, cond::pluginCategory());

namespace cond {
  const char* pluginCategory() { return "CondProductResolverFactory"; }
}  // namespace cond
