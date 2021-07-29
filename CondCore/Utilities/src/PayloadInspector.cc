#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/ConnectionPool.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"

#include <sstream>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/embed.h>

namespace py = pybind11;

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

    PlotBase::PlotBase()
        : m_plotAnnotations(),
          m_inputParams(),
          m_tagNames(),
          m_tagBoundaries(),
          m_tagIovs(),
          m_inputParamValues(),
          m_data("") {}

    void PlotBase::addInputParam(const std::string& paramName) {
      // maybe add a check for existing params - returning an Exception when found...
      m_inputParams.insert(paramName);
    }

    std::string PlotBase::payloadType() const { return m_plotAnnotations.get(PlotAnnotations::PAYLOAD_TYPE_K); }

    std::string PlotBase::title() const { return m_plotAnnotations.get(PlotAnnotations::TITLE_K); }

    std::string PlotBase::type() const { return m_plotAnnotations.get(PlotAnnotations::PLOT_TYPE_K); }

    bool PlotBase::isSingleIov() const { return m_plotAnnotations.singleIov; }

    unsigned int PlotBase::ntags() const { return m_plotAnnotations.ntags; }

    bool PlotBase::isTwoTags() const { return m_plotAnnotations.ntags == 2; }

    py::list PlotBase::inputParams() const {
      py::list tmp;
      for (const auto& ip : m_inputParams) {
        tmp.append(ip);
      }
      return tmp;
    }

     void PlotBase::setInputParamValues(const py::dict& values) {
       for(auto item : values){
          std::string key = item.first.cast<std::string>();
          std::string val = item.second.cast<std::string>();
          m_inputParamValues.insert(std::make_pair(key, val));            
       }
    }

    std::string PlotBase::data() const { return m_data; }

    bool PlotBase::process(const std::string& connectionString, const py::list& tagsWithTimeBoundaries) {
      size_t nt = tagsWithTimeBoundaries.size();
      bool ret = false;
      py::object obj1,obj2,obj3,obj4;
      if (nt) {
        std::vector<std::tuple<std::string, cond::Time_t, cond::Time_t> > tags;
        tags.resize(nt);
        for (size_t i = 0; i < nt; i++) {
          obj1=tagsWithTimeBoundaries[i];
          py::tuple entry = obj1.cast<py::tuple>();
          obj2=entry[0];
          obj3=entry[1];
          obj4=entry[2];
          std::string tagName = obj2.cast<std::string>();
          std::string time0s = obj3.cast<std::string>();
          std::string time1s = obj4.cast<std::string>();
          cond::Time_t time0 = boost::lexical_cast<cond::Time_t>(time0s);
          cond::Time_t time1 = boost::lexical_cast<cond::Time_t>(time1s);
          tags[i] = std::make_tuple(tagName, time0, time1);
        }
        ret = exec_process(connectionString, tags);
      }
      return ret;
    }

    bool PlotBase::exec_process(
        const std::string& connectionString,
        const std::vector<std::tuple<std::string, cond::Time_t, cond::Time_t> >& tagsWithTimeBoundaries) {
      m_tagNames.clear();
      m_tagBoundaries.clear();
      m_tagIovs.clear();
      init();

      std::vector<edm::ParameterSet> psets;
      edm::ParameterSet pSet;
      pSet.addParameter("@service_type", std::string("SiteLocalConfigService"));
      psets.push_back(pSet);
      static const edm::ServiceToken services(edm::ServiceRegistry::createSet(psets));
      static const edm::ServiceRegistry::Operate operate(services);
      bool ret = false;
      size_t nt = tagsWithTimeBoundaries.size();
      if (nt) {
        cond::persistency::ConnectionPool connection;
        m_dbSession = connection.createSession(connectionString);
        m_dbSession.transaction().start();
        m_tagNames.resize(nt);
        m_tagBoundaries.resize(nt);
        m_tagIovs.resize(nt);
        for (size_t i = 0; i < nt; i++) {
          const std::string& tagName = std::get<0>(tagsWithTimeBoundaries[i]);
          cond::Time_t time0 = std::get<1>(tagsWithTimeBoundaries[i]);
          cond::Time_t time1 = std::get<2>(tagsWithTimeBoundaries[i]);
          m_tagNames[i] = tagName;
          m_tagBoundaries[i] = std::make_pair(time0, time1);
          auto proxy = m_dbSession.readIov(tagName);
          proxy.selectRange(time0, time1, m_tagIovs[i]);
        }
        m_data = processData();
        m_dbSession.transaction().commit();
        ret = true;
      }
      return ret;
    }

    void PlotBase::init() {}

    std::string PlotBase::processData() { return ""; }

    cond::Tag_t PlotBase::getTagInfo(const std::string& tag) {
      cond::Tag_t info = m_dbSession.readIov(tag).tagInfo();
      return info;
    }

    const std::map<std::string, std::string>& PlotBase::inputParamValues() const { return m_inputParamValues; }

    cond::persistency::Session PlotBase::dbSession() { return m_dbSession; }

  }  // namespace payloadInspector

}  // namespace cond
