#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/ConnectionPool.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"

#include <sstream>
#include <iostream>
#include <boost/python/extract.hpp>
#include <boost/python/tuple.hpp>

namespace cond {

  namespace payloadInspector {

    PlotAnnotations::PlotAnnotations() : m() {}

    std::string PlotAnnotations::get(const std::string& key) const {
      std::string ret("");
      auto im = m.find(key);
      if (im != m.end())
        ret = im->second;
      return ret;
    }

    constexpr const char* const ModuleVersion::label;

    PlotBase::PlotBase() : m_plotAnnotations(), m_inputParams(), m_data(""), m_inputParamValues() {}

    void PlotBase::addInputParam(const std::string& paramName) {
      // maybe add a check for existing params - returning an Exception when found...
      m_inputParams.insert(paramName);
    }

    std::string PlotBase::payloadType() const { return m_plotAnnotations.get(PlotAnnotations::PAYLOAD_TYPE_K); }

    std::string PlotBase::title() const { return m_plotAnnotations.get(PlotAnnotations::TITLE_K); }

    std::string PlotBase::type() const { return m_plotAnnotations.get(PlotAnnotations::PLOT_TYPE_K); }

    bool PlotBase::isSingleIov() const { return m_plotAnnotations.singleIov; }

    bool PlotBase::isTwoTags() const { return m_plotAnnotations.twoTags; }

    boost::python::list PlotBase::inputParams() const {
      boost::python::list tmp;
      for (auto ip : m_inputParams) {
        tmp.append(ip);
      }
      return tmp;
    }

    void PlotBase::setInputParamValues(const boost::python::dict& values) {
      for (auto ip : m_inputParams) {
        if (values.has_key(ip)) {
          std::string val = boost::python::extract<std::string>(values.get(ip));
          m_inputParamValues.insert(std::make_pair(ip, val));
        }
      }
    }

    std::string PlotBase::data() const { return m_data; }

    bool PlotBase::process(const std::string& connectionString,
                           const std::string& tag,
                           const std::string&,
                           cond::Time_t begin,
                           cond::Time_t end) {
      init();

      std::vector<edm::ParameterSet> psets;
      edm::ParameterSet pSet;
      pSet.addParameter("@service_type", std::string("SiteLocalConfigService"));
      psets.push_back(pSet);
      static const edm::ServiceToken services(edm::ServiceRegistry::createSet(psets));
      static const edm::ServiceRegistry::Operate operate(services);

      m_tag0 = tag;
      //m_tagTimeType = cond::time::timeTypeFromName(timeType);
      cond::persistency::ConnectionPool connection;
      m_dbSession = connection.createSession(connectionString);
      m_dbSession.transaction().start();
      std::vector<std::tuple<cond::Time_t, cond::Hash> > iovs;
      m_dbSession.getIovRange(tag, begin, end, iovs);
      m_data = processData(iovs);
      m_dbSession.transaction().commit();
      // fixme...
      return true;
    }

    bool PlotBase::processTwoTags(const std::string& connectionString,
                                  const std::string& tag0,
                                  const std::string& tag1,
                                  cond::Time_t time0,
                                  cond::Time_t time1) {
      init();

      std::vector<edm::ParameterSet> psets;
      edm::ParameterSet pSet;
      pSet.addParameter("@service_type", std::string("SiteLocalConfigService"));
      psets.push_back(pSet);
      static const edm::ServiceToken services(edm::ServiceRegistry::createSet(psets));
      static const edm::ServiceRegistry::Operate operate(services);

      m_tag0 = tag0;
      m_tag1 = tag1;
      cond::persistency::ConnectionPool connection;
      m_dbSession = connection.createSession(connectionString);
      m_dbSession.transaction().start();
      std::vector<std::tuple<cond::Time_t, cond::Hash> > iovs;
      m_dbSession.getIovRange(tag0, time0, time0, iovs);
      m_dbSession.getIovRange(tag1, time1, time1, iovs);
      m_data = processData(iovs);
      m_dbSession.transaction().commit();
      // fixme...
      return true;
    }

    void PlotBase::init() {}

    std::string PlotBase::processData(const std::vector<std::tuple<cond::Time_t, cond::Hash> >&) { return ""; }

    void PlotBase::setSingleIov(bool flag) { m_plotAnnotations.singleIov = flag; }

    void PlotBase::setTwoTags(bool flag) {
      m_plotAnnotations.twoTags = flag;
      if (flag)
        m_plotAnnotations.singleIov = flag;
    }

    cond::Tag_t PlotBase::getTagInfo(const std::string& tag) {
      cond::Tag_t info;
      m_dbSession.getTagInfo(tag, info);
      return info;
    }

    const std::map<std::string, std::string>& PlotBase::inputParamValues() const { return m_inputParamValues; }

    cond::persistency::Session PlotBase::dbSession() { return m_dbSession; }

  }  // namespace payloadInspector

}  // namespace cond
