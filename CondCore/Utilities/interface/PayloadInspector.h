#ifndef CondCore_Utilities_PayloadInspector_h
#define CondCore_Utilities_PayloadInspector_h

#include "CondCore/CondDB/interface/Utils.h"
#include "CondCore/CondDB/interface/Session.h"
#include "CondCore/CondDB/interface/Exception.h"
#include <iostream>

#include <string>
#include <tuple>
#include <vector>
#include <type_traits>

#include "FWCore/Utilities/interface/GlobalIdentifier.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <boost/python/list.hpp>
#include <boost/python/dict.hpp>
#include <boost/python/tuple.hpp>

namespace PI {
  inline boost::python::list mk_input(const std::string& tagName, cond::Time_t start, cond::Time_t end) {
    boost::python::list ret;
    ret.append(boost::python::make_tuple(tagName, std::to_string(start), std::to_string(end)));
    return ret;
  }
  inline boost::python::list mk_input(const std::string& tagName0,
                                      cond::Time_t start0,
                                      cond::Time_t end0,
                                      const std::string& tagName1,
                                      cond::Time_t start1,
                                      cond::Time_t end1) {
    boost::python::list ret;
    ret.append(boost::python::make_tuple(tagName0, std::to_string(start0), std::to_string(end0)));
    ret.append(boost::python::make_tuple(tagName1, std::to_string(start1), std::to_string(end1)));
    return ret;
  }
}  // namespace PI

namespace cond {

  namespace payloadInspector {

    // Metadata dictionary
    struct PlotAnnotations {
      static constexpr const char* const PLOT_TYPE_K = "type";
      static constexpr const char* const TITLE_K = "title";
      static constexpr const char* const PAYLOAD_TYPE_K = "payload_type";
      static constexpr const char* const INFO_K = "info";
      static constexpr const char* const XAXIS_K = "x_label";
      static constexpr const char* const YAXIS_K = "y_label";
      static constexpr const char* const ZAXIS_K = "z_label";
      PlotAnnotations();
      std::string get(const std::string& key) const;
      std::map<std::string, std::string> m;
      int ntags = 1;
      bool twoTags = false;
      bool singleIov = false;
    };

    static const char* const JSON_FORMAT_VERSION = "1.0";

    // Serialize functions
    template <typename V>
    std::string serializeValue(const std::string& entryLabel, const V& value) {
      std::stringstream ss;

      // N.B.:
      //  This hack is to output a line to stringstream only in case the
      //  return type of getFromPayload is a std::pair<bool, float>
      //  and the bool is true. This allows to control which points should
      //  enter the trend and which should not.

      if constexpr (std::is_same_v<V, std::pair<bool, float>>) {
        if (value.first) {
          ss << "\"" << entryLabel << "\":" << value.second;
        }
      } else if constexpr (std::is_same_v<V, double>) {
        ss.precision(0);
        ss << "\"" << entryLabel << "\":" << std::fixed << value;
      } else {
        ss << "\"" << entryLabel << "\":" << value;
      }
      return ss.str();
    }

    template <>
    inline std::string serializeValue(const std::string& entryLabel, const std::string& value) {
      std::stringstream ss;
      ss << "\"" << entryLabel << "\":\"" << value << "\"";
      return ss.str();
    }

    // Specialization for the multi-values coordinates ( to support the combined time+runlumi abscissa )
    template <typename V>
    std::string serializeValue(const std::string& entryLabel, const std::tuple<V, std::string>& value) {
      std::stringstream ss;
      ss << serializeValue(entryLabel, std::get<0>(value));
      ss << ", ";
      ss << serializeValue(entryLabel + "_label", std::get<1>(value));
      return ss.str();
    }

    // Specialization for the error bars
    template <typename V>
    std::string serializeValue(const std::string& entryLabel, const std::pair<V, V>& value) {
      std::stringstream ss;
      ss << serializeValue(entryLabel, value.first);
      ss << ", ";
      ss << serializeValue(entryLabel + "_err", value.second);
      return ss.str();
    }

    inline std::string serializeAnnotations(const PlotAnnotations& annotations) {
      std::stringstream ss;
      ss << "\"version\": \"" << JSON_FORMAT_VERSION << "\",";
      ss << "\"annotations\": {";
      bool first = true;
      for (const auto& a : annotations.m) {
        if (!first)
          ss << ",";
        ss << "\"" << a.first << "\":\"" << a.second << "\"";
        first = false;
      }
      ss << "}";
      return ss.str();
    }

    template <typename X, typename Y>
    std::string serialize(const PlotAnnotations& annotations, const std::vector<std::tuple<X, Y>>& data) {
      // prototype implementation...
      std::stringstream ss;
      ss << "{";
      ss << serializeAnnotations(annotations);
      ss << ",";
      ss << "\"data\": [";
      bool first = true;
      for (auto d : data) {
        auto serializedX = serializeValue("x", std::get<0>(d));
        auto serializedY = serializeValue("y", std::get<1>(d));

        // N.B.:
        //  we output to JSON only if the stringstream
        //  from serializeValue is not empty

        if (!serializedY.empty()) {
          if (!first) {
            ss << ",";
          }
          ss << " { " << serializedX << ", " << serializedY << " }";
          first = false;
        }
      }
      ss << "]";
      ss << "}";
      return ss.str();
    }

    template <typename X, typename Y, typename Z>
    std::string serialize(const PlotAnnotations& annotations, const std::vector<std::tuple<X, Y, Z>>& data) {
      // prototype implementation...
      std::stringstream ss;
      ss << "{";
      ss << serializeAnnotations(annotations);
      ss << ",";
      ss << "\"data\": [";
      bool first = true;
      for (auto d : data) {
        if (!first)
          ss << ",";
        ss << " { " << serializeValue("x", std::get<0>(d)) << ", " << serializeValue("y", std::get<1>(d)) << ", "
           << serializeValue("z", std::get<2>(d)) << " }";
        first = false;
      }
      ss << "]";
      ss << "}";
      return ss.str();
    }

    inline std::string serialize(const PlotAnnotations& annotations, const std::string& imageFileName) {
      std::stringstream ss;
      ss << "{";
      ss << serializeAnnotations(annotations);
      ss << ",";
      ss << "\"file\": \"" << imageFileName << "\"";
      ss << "}";
      return ss.str();
    }

    struct ModuleVersion {
      static constexpr const char* const label = "2.0";
    };

    struct TagReference {
      TagReference(const std::string& n,
                   const std::pair<cond::Time_t, cond::Time_t>& b,
                   const std::vector<std::tuple<cond::Time_t, cond::Hash>>& i)
          : name(n), boundary(b), iovs(i) {}
      TagReference(const TagReference& rhs) : name(rhs.name), boundary(rhs.boundary), iovs(rhs.iovs) {}
      const std::string& name;
      const std::pair<cond::Time_t, cond::Time_t>& boundary;
      const std::vector<std::tuple<cond::Time_t, cond::Hash>>& iovs;
    };

    // Base class, factorizing the functions exposed in the python interface
    class PlotBase {
    public:
      PlotBase();
      virtual ~PlotBase() = default;
      // required in the browser to find corresponding tags
      std::string payloadType() const;

      // required in the browser
      std::string title() const;

      // required in the browser
      std::string type() const;

      // required in the browser
      unsigned int ntags() const;

      //TBRemoved
      bool isTwoTags() const;

      // required in the browser
      bool isSingleIov() const;

      // required in the browser
      boost::python::list inputParams() const;

      // required in the browser
      void setInputParamValues(const boost::python::dict& values);

      // returns the json file with the plot data
      std::string data() const;

      // triggers the processing producing the plot
      bool process(const std::string& connectionString, const boost::python::list& tagsWithTimeBoundaries);

      // called by the above method - to be used in C++ unit tests...
      bool exec_process(const std::string& connectionString,
                        const std::vector<std::tuple<std::string, cond::Time_t, cond::Time_t>>& tagsWithTimeBoundaries);

      // not exposed in python:
      // called internally in process()
      virtual void init();

      // not exposed in python:
      // called internally in process()
      //virtual std::string processData(const std::vector<std::tuple<cond::Time_t, cond::Hash> >& iovs);
      virtual std::string processData();

      //
      void addInputParam(const std::string& paramName);

      // access to the fetch function of the configured reader, to be used in the processData implementations
      template <typename PayloadType>
      std::shared_ptr<PayloadType> fetchPayload(const cond::Hash& payloadHash) {
        return m_dbSession.fetchPayload<PayloadType>(payloadHash);
      }

      cond::Tag_t getTagInfo(const std::string& tag);

      template <int index>
      TagReference getTag() {
        size_t sz = m_tagNames.size();
        if (sz == 0 || index >= sz) {
          cond::throwException("Index out of range", "PlotBase::getTag()");
        }
        return TagReference(m_tagNames[index], m_tagBoundaries[index], m_tagIovs[index]);
      }

      const std::map<std::string, std::string>& inputParamValues() const;

      // access to the underlying db session
      cond::persistency::Session dbSession();

    protected:
      // possibly shared with the derived classes
      PlotAnnotations m_plotAnnotations;
      std::set<std::string> m_inputParams;
      std::vector<std::string> m_tagNames;
      std::vector<std::pair<cond::Time_t, cond::Time_t>> m_tagBoundaries;
      std::vector<std::vector<std::tuple<cond::Time_t, cond::Hash>>> m_tagIovs;
      std::map<std::string, std::string> m_inputParamValues;

    private:
      // this stuff should not be modified...
      cond::persistency::Session m_dbSession;
      std::string m_data = "";
    };

    enum IOVMultiplicity { UNSPECIFIED_IOV = 0, MULTI_IOV = 1, SINGLE_IOV = 2 };

    inline void setAnnotations(
        const std::string& type, const std::string& title, IOVMultiplicity IOV_M, int NTAGS, PlotAnnotations& target) {
      target.m[PlotAnnotations::PLOT_TYPE_K] = type;
      target.m[PlotAnnotations::TITLE_K] = title;
      target.ntags = NTAGS;
      target.singleIov = (IOV_M == SINGLE_IOV);
    }

    template <IOVMultiplicity IOV_M, int NTAGS>
    class PlotImpl : public PlotBase {
    public:
      PlotImpl(const std::string& type, const std::string& title) : PlotBase() {
        setAnnotations(type, title, IOV_M, NTAGS, m_plotAnnotations);
      }
      ~PlotImpl() override = default;

      virtual std::string serializeData() = 0;

      std::string processData() override {
        fill();
        return serializeData();
      }

      virtual bool fill() = 0;
    };

    // specialisations
    template <int NTAGS>
    class PlotImpl<UNSPECIFIED_IOV, NTAGS> : public PlotBase {
    public:
      PlotImpl(const std::string& type, const std::string& title) : PlotBase() {
        setAnnotations(type, title, MULTI_IOV, NTAGS, m_plotAnnotations);
      }
      ~PlotImpl() override = default;

      virtual std::string serializeData() = 0;

      std::string processData() override {
        fill();
        return serializeData();
      }

      virtual bool fill() = 0;

      void setSingleIov(bool flag) { m_plotAnnotations.singleIov = flag; }
    };

    template <>
    class PlotImpl<MULTI_IOV, 0> : public PlotBase {
    public:
      PlotImpl(const std::string& type, const std::string& title) : PlotBase() {
        setAnnotations(type, title, MULTI_IOV, 1, m_plotAnnotations);
      }
      ~PlotImpl() override = default;

      virtual std::string serializeData() = 0;

      std::string processData() override {
        fill();
        return serializeData();
      }

      virtual bool fill() = 0;

      void setTwoTags(bool flag) {
        if (flag)
          m_plotAnnotations.ntags = 2;
        else
          m_plotAnnotations.ntags = 1;
      }
    };

    template <>
    class PlotImpl<SINGLE_IOV, 0> : public PlotBase {
    public:
      PlotImpl(const std::string& type, const std::string& title) : PlotBase() {
        setAnnotations(type, title, SINGLE_IOV, 1, m_plotAnnotations);
      }
      ~PlotImpl() override = default;

      virtual std::string serializeData() = 0;

      std::string processData() override {
        fill();
        return serializeData();
      }

      virtual bool fill() = 0;

      void setTwoTags(bool flag) {
        if (flag)
          m_plotAnnotations.ntags = 2;
        else
          m_plotAnnotations.ntags = 1;
      }
    };

    template <>
    class PlotImpl<UNSPECIFIED_IOV, 0> : public PlotBase {
    public:
      PlotImpl(const std::string& type, const std::string& title) : PlotBase() {
        setAnnotations(type, title, MULTI_IOV, 1, m_plotAnnotations);
      }
      ~PlotImpl() override = default;

      virtual std::string serializeData() = 0;

      std::string processData() override {
        fill();
        return serializeData();
      }

      virtual bool fill() {
        std::vector<std::tuple<cond::Time_t, cond::Hash>> theIovs = PlotBase::getTag<0>().iovs;
        if (m_plotAnnotations.ntags == 2) {
          auto tag2iovs = PlotBase::getTag<1>().iovs;
          size_t oldSize = theIovs.size();
          size_t newSize = oldSize + tag2iovs.size();
          theIovs.resize(newSize);
          for (size_t i = 0; i < tag2iovs.size(); i++) {
            theIovs[i + oldSize] = tag2iovs[i];
          }
        }
        return fill(theIovs);
      }

      virtual bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash>>& iovs) { return false; }

      void setSingleIov(bool flag) {
        m_plotAnnotations.singleIov = flag;
        m_singleIovSet = true;
      }

      void setTwoTags(bool flag) {
        if (flag) {
          m_plotAnnotations.ntags = 2;
          if (!m_singleIovSet)
            m_plotAnnotations.singleIov = true;
        } else
          m_plotAnnotations.ntags = 1;
      }

      bool m_singleIovSet = false;
    };

    template <>
    class PlotImpl<UNSPECIFIED_IOV, 1> : public PlotBase {
    public:
      PlotImpl(const std::string& type, const std::string& title) : PlotBase() {
        setAnnotations(type, title, MULTI_IOV, 1, m_plotAnnotations);
      }
      ~PlotImpl() override = default;

      virtual std::string serializeData() = 0;

      std::string processData() override {
        fill();
        return serializeData();
      }

      virtual bool fill() {
        std::vector<std::tuple<cond::Time_t, cond::Hash>> theIovs = PlotBase::getTag<0>().iovs;
        if (m_plotAnnotations.ntags == 2) {
          auto tag2iovs = PlotBase::getTag<1>().iovs;
          size_t oldSize = theIovs.size();
          size_t newSize = oldSize + tag2iovs.size();
          theIovs.resize(newSize);
          for (size_t i = 0; i < tag2iovs.size(); i++) {
            theIovs[i + oldSize] = tag2iovs[i];
          }
        }
        return fill(theIovs);
      }

      virtual bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash>>& iovs) { return false; }

      void setSingleIov(bool flag) { m_plotAnnotations.singleIov = flag; }
    };

    // Concrete plot-types implementations
    template <typename PayloadType, typename X, typename Y, IOVMultiplicity IOV_M = UNSPECIFIED_IOV, int NTAGS = 0>
    class Plot2D : public PlotImpl<IOV_M, NTAGS> {
    public:
      typedef PlotImpl<IOV_M, NTAGS> Base;
      Plot2D(const std::string& type, const std::string& title, const std::string xLabel, const std::string& yLabel)
          : Base(type, title), m_plotData() {
        Base::m_plotAnnotations.m[PlotAnnotations::XAXIS_K] = xLabel;
        Base::m_plotAnnotations.m[PlotAnnotations::YAXIS_K] = yLabel;
        Base::m_plotAnnotations.m[PlotAnnotations::PAYLOAD_TYPE_K] = cond::demangledName(typeid(PayloadType));
      }
      ~Plot2D() override = default;
      std::string serializeData() override { return serialize(Base::m_plotAnnotations, m_plotData); }

      std::shared_ptr<PayloadType> fetchPayload(const cond::Hash& payloadHash) {
        return PlotBase::fetchPayload<PayloadType>(payloadHash);
      }

    protected:
      std::vector<std::tuple<X, Y>> m_plotData;
    };

    template <typename PayloadType,
              typename X,
              typename Y,
              typename Z,
              IOVMultiplicity IOV_M = UNSPECIFIED_IOV,
              int NTAGS = 0>
    class Plot3D : public PlotImpl<IOV_M, NTAGS> {
    public:
      typedef PlotImpl<IOV_M, NTAGS> Base;
      Plot3D(const std::string& type,
             const std::string& title,
             const std::string xLabel,
             const std::string& yLabel,
             const std::string& zLabel)
          : Base(type, title), m_plotData() {
        Base::m_plotAnnotations.m[PlotAnnotations::XAXIS_K] = xLabel;
        Base::m_plotAnnotations.m[PlotAnnotations::YAXIS_K] = yLabel;
        Base::m_plotAnnotations.m[PlotAnnotations::ZAXIS_K] = zLabel;
        Base::m_plotAnnotations.m[PlotAnnotations::PAYLOAD_TYPE_K] = cond::demangledName(typeid(PayloadType));
      }
      ~Plot3D() override = default;
      std::string serializeData() override { return serialize(Base::m_plotAnnotations, m_plotData); }

      std::shared_ptr<PayloadType> fetchPayload(const cond::Hash& payloadHash) {
        return PlotBase::fetchPayload<PayloadType>(payloadHash);
      }

    protected:
      std::vector<std::tuple<X, Y, Z>> m_plotData;
    };

    template <typename PayloadType, typename Y>
    class HistoryPlot : public Plot2D<PayloadType, unsigned long long, Y, MULTI_IOV, 1> {
    public:
      typedef Plot2D<PayloadType, unsigned long long, Y, MULTI_IOV, 1> Base;

      HistoryPlot(const std::string& title, const std::string& yLabel) : Base("History", title, "iov_since", yLabel) {}

      ~HistoryPlot() override = default;

      bool fill() override {
        auto tag = PlotBase::getTag<0>();
        for (auto iov : tag.iovs) {
          std::shared_ptr<PayloadType> payload = Base::fetchPayload(std::get<1>(iov));
          if (payload.get()) {
            Y value = getFromPayload(*payload);
            Base::m_plotData.push_back(std::make_tuple(std::get<0>(iov), value));
          }
        }
        return true;
      }
      virtual Y getFromPayload(PayloadType& payload) = 0;
    };

    template <typename PayloadType, typename Y>
    class RunHistoryPlot : public Plot2D<PayloadType, std::tuple<float, std::string>, Y, MULTI_IOV, 1> {
    public:
      typedef Plot2D<PayloadType, std::tuple<float, std::string>, Y, MULTI_IOV, 1> Base;

      RunHistoryPlot(const std::string& title, const std::string& yLabel)
          : Base("RunHistory", title, "iov_since", yLabel) {}

      ~RunHistoryPlot() override = default;

      bool fill() override {
        auto tag = PlotBase::getTag<0>();
        // for the lumi iovs we need to count the number of lumisections in every runs
        std::map<cond::Time_t, unsigned int> runs;

        cond::Tag_t tagInfo = Base::getTagInfo(tag.name);
        if (tagInfo.timeType == cond::lumiid) {
          for (auto iov : tag.iovs) {
            unsigned int run = std::get<0>(iov) >> 32;
            auto it = runs.find(run);
            if (it == runs.end())
              it = runs.insert(std::make_pair(run, 0)).first;
            it->second++;
          }
        }
        unsigned int currentRun = 0;
        float lumiIndex = 0;
        unsigned int lumiSize = 0;
        unsigned int rind = 0;
        float ind = 0;
        std::string label("");
        for (auto iov : tag.iovs) {
          unsigned long long since = std::get<0>(iov);
          // for the lumi iovs we squeeze the lumi section available in the constant run 'slot' of witdth=1
          if (tagInfo.timeType == cond::lumiid) {
            unsigned int run = since >> 32;
            unsigned int lumi = since & 0xFFFFFFFF;
            if (run != currentRun) {
              rind++;
              lumiIndex = 0;
              auto it = runs.find(run);
              if (it == runs.end()) {
                // it should never happen
                return false;
              }
              lumiSize = it->second;
            } else {
              lumiIndex++;
            }
            ind = rind + (lumiIndex / lumiSize);
            label = std::to_string(run) + " : " + std::to_string(lumi);
            currentRun = run;
          } else {
            ind++;
            // for the timestamp based iovs, it does not really make much sense to use this plot...
            if (tagInfo.timeType == cond::timestamp) {
              boost::posix_time::ptime t = cond::time::to_boost(since);
              label = boost::posix_time::to_simple_string(t);
            } else {
              label = std::to_string(since);
            }
          }
          std::shared_ptr<PayloadType> payload = Base::fetchPayload(std::get<1>(iov));
          if (payload.get()) {
            Y value = getFromPayload(*payload);
            Base::m_plotData.push_back(std::make_tuple(std::make_tuple(ind, label), value));
          }
        }
        return true;
      }

      virtual Y getFromPayload(PayloadType& payload) = 0;
    };

    template <typename PayloadType, typename Y>
    class TimeHistoryPlot : public Plot2D<PayloadType, std::tuple<unsigned long long, std::string>, Y, MULTI_IOV, 1> {
    public:
      typedef Plot2D<PayloadType, std::tuple<unsigned long long, std::string>, Y, MULTI_IOV, 1> Base;

      TimeHistoryPlot(const std::string& title, const std::string& yLabel)
          : Base("TimeHistory", title, "iov_since", yLabel) {}
      ~TimeHistoryPlot() override = default;

      bool fill() override {
        auto tag = PlotBase::getTag<0>();
        cond::persistency::RunInfoProxy runInfo;

        cond::Tag_t tagInfo = Base::getTagInfo(tag.name);
        if (tagInfo.timeType == cond::lumiid || tagInfo.timeType == cond::runnumber) {
          cond::Time_t min = std::get<0>(tag.iovs.front());
          cond::Time_t max = std::get<0>(tag.iovs.back());
          if (tagInfo.timeType == cond::lumiid) {
            min = min >> 32;
            max = max >> 32;
          }
          runInfo = Base::dbSession().getRunInfo(min, max);
        }
        for (auto iov : tag.iovs) {
          cond::Time_t since = std::get<0>(iov);
          boost::posix_time::ptime time;
          std::string label("");
          if (tagInfo.timeType == cond::lumiid || tagInfo.timeType == cond::runnumber) {
            unsigned int nlumi = since & 0xFFFFFFFF;
            if (tagInfo.timeType == cond::lumiid)
              since = since >> 32;
            label = std::to_string(since);
            auto it = runInfo.find(since);
            if (it == runInfo.end()) {
              // this should never happen...
              return false;
            }
            time = (*it).start;
            // add the lumi sections...
            if (tagInfo.timeType == cond::lumiid) {
              time += boost::posix_time::seconds(cond::time::SECONDS_PER_LUMI * nlumi);
              label += (" : " + std::to_string(nlumi));
            }
          } else if (tagInfo.timeType == cond::timestamp) {
            time = cond::time::to_boost(since);
            label = boost::posix_time::to_simple_string(time);
          }
          std::shared_ptr<PayloadType> payload = Base::fetchPayload(std::get<1>(iov));
          if (payload.get()) {
            Y value = getFromPayload(*payload);
            Base::m_plotData.push_back(std::make_tuple(std::make_tuple(cond::time::from_boost(time), label), value));
          }
        }
        return true;
      }

      virtual Y getFromPayload(PayloadType& payload) = 0;
    };

    template <typename PayloadType, typename X, typename Y>
    class ScatterPlot : public Plot2D<PayloadType, X, Y, MULTI_IOV, 1> {
    public:
      typedef Plot2D<PayloadType, X, Y, MULTI_IOV, 1> Base;
      // the x axis label will be overwritten by the plot rendering application
      ScatterPlot(const std::string& title, const std::string& xLabel, const std::string& yLabel)
          : Base("Scatter", title, xLabel, yLabel) {}
      ~ScatterPlot() override = default;

      bool fill() override {
        auto tag = PlotBase::getTag<0>();
        for (auto iov : tag.iovs) {
          std::shared_ptr<PayloadType> payload = Base::fetchPayload(std::get<1>(iov));
          if (payload.get()) {
            std::tuple<X, Y> value = getFromPayload(*payload);
            Base::m_plotData.push_back(value);
          }
        }
        return true;
      }

      virtual std::tuple<X, Y> getFromPayload(PayloadType& payload) = 0;
    };

    //
    template <typename AxisType, typename PayloadType, IOVMultiplicity IOV_M = UNSPECIFIED_IOV>
    class Histogram1 : public Plot2D<PayloadType, AxisType, AxisType, IOV_M, 1> {
    public:
      typedef Plot2D<PayloadType, AxisType, AxisType, IOV_M, 1> Base;
      // naive implementation, essentially provided as an example...
      Histogram1(const std::string& title,
                 const std::string& xLabel,
                 size_t nbins,
                 float min,
                 float max,
                 const std::string& yLabel = "entries")
          : Base("Histo1D", title, xLabel, yLabel), m_nbins(nbins), m_min(min), m_max(max) {}

      //
      void init() override {
        if (m_nbins < 1) {
          edm::LogError("payloadInspector::Histogram1D()")
              << " trying to book an histogram with less then 1 bin!" << std::endl;
        }

        if (m_min > m_max) {
          edm::LogError("payloadInspector::Histogram1D()")
              << " trying to book an histogram with minimum " << m_min << "> maximum" << m_max << " !" << std::endl;
        }

        Base::m_plotData.clear();
        float binSize = (m_max - m_min) / m_nbins;
        if (binSize > 0) {
          m_binSize = binSize;
          Base::m_plotData.resize(m_nbins);
          for (size_t i = 0; i < m_nbins; i++) {
            Base::m_plotData[i] = std::make_tuple(m_min + i * m_binSize, 0);
          }
        }
      }

      // to be used to fill the histogram!
      void fillWithValue(AxisType value, AxisType weight = 1) {
        // ignoring underflow/overflows ( they can be easily added - the total entries as well  )
        if (!Base::m_plotData.empty() && (value < m_max) && (value >= m_min)) {
          size_t ibin = (value - m_min) / m_binSize;
          std::get<1>(Base::m_plotData[ibin]) += weight;
        }
      }

      // to be used to fill the histogram!
      void fillWithBinAndValue(size_t bin, AxisType weight = 1) {
        if (bin < Base::m_plotData.size()) {
          std::get<1>(Base::m_plotData[bin]) = weight;
        }
      }

      // this one can ( and in general should ) be overridden - the implementation should use fillWithValue
      bool fill() override {
        auto tag = PlotBase::getTag<0>();
        for (auto iov : tag.iovs) {
          std::shared_ptr<PayloadType> payload = Base::fetchPayload(std::get<1>(iov));
          if (payload.get()) {
            AxisType value = getFromPayload(*payload);
            fillWithValue(value);
          }
        }
        return true;
      }

      // implement this one if you use the default fill implementation, otherwise ignore it...
      virtual AxisType getFromPayload(PayloadType& payload) { return 0; }

    private:
      float m_binSize = 0;
      size_t m_nbins;
      float m_min;
      float m_max;
    };

    // clever way to reduce the number of templated arguments
    // see https://stackoverflow.com/questions/3881633/reducing-number-of-template-arguments-for-class
    // for reference

    template <typename PayloadType, IOVMultiplicity IOV_M = UNSPECIFIED_IOV>
    using Histogram1D = Histogram1<float, PayloadType, UNSPECIFIED_IOV>;

    template <typename PayloadType, IOVMultiplicity IOV_M = UNSPECIFIED_IOV>
    using Histogram1DD = Histogram1<double, PayloadType, UNSPECIFIED_IOV>;

    //
    template <typename PayloadType, IOVMultiplicity IOV_M = UNSPECIFIED_IOV>
    class Histogram2D : public Plot3D<PayloadType, float, float, float, IOV_M, 1> {
    public:
      typedef Plot3D<PayloadType, float, float, float, IOV_M, 1> Base;
      // naive implementation, essentially provided as an example...
      Histogram2D(const std::string& title,
                  const std::string& xLabel,
                  size_t nxbins,
                  float xmin,
                  float xmax,
                  const std::string& yLabel,
                  size_t nybins,
                  float ymin,
                  float ymax)
          : Base("Histo2D", title, xLabel, yLabel, "entries"),
            m_nxbins(nxbins),
            m_xmin(xmin),
            m_xmax(xmax),
            m_nybins(nybins),
            m_ymin(ymin),
            m_ymax(ymax) {}

      //
      void init() override {
        // some protections
        if ((m_nxbins < 1) || (m_nybins < 1)) {
          edm::LogError("payloadInspector::Histogram2D()")
              << " trying to book an histogram with less then 1 bin!" << std::endl;
        }

        if (m_xmin > m_xmax) {
          edm::LogError("payloadInspector::Histogram2D()") << " trying to book an histogram with x-minimum " << m_xmin
                                                           << "> x-maximum" << m_xmax << " !" << std::endl;
        }

        if (m_ymin > m_ymax) {
          edm::LogError("payloadInspector::Histogram2D()") << " trying to book an histogram with y-minimum " << m_ymin
                                                           << "> y-maximum" << m_ymax << " !" << std::endl;
        }

        Base::m_plotData.clear();
        float xbinSize = (m_xmax - m_xmin) / m_nxbins;
        float ybinSize = (m_ymax - m_ymin) / m_nybins;
        if (xbinSize > 0 && ybinSize > 0) {
          m_xbinSize = xbinSize;
          m_ybinSize = ybinSize;
          Base::m_plotData.resize(m_nxbins * m_nybins);
          for (size_t i = 0; i < m_nybins; i++) {
            for (size_t j = 0; j < m_nxbins; j++) {
              Base::m_plotData[i * m_nxbins + j] = std::make_tuple(m_xmin + j * m_xbinSize, m_ymin + i * m_ybinSize, 0);
            }
          }
        }
      }

      // to be used to fill the histogram!
      void fillWithValue(float xvalue, float yvalue, float weight = 1) {
        // ignoring underflow/overflows ( they can be easily added - the total entries as well )
        if (!Base::m_plotData.empty() && xvalue < m_xmax && xvalue >= m_xmin && yvalue < m_ymax && yvalue >= m_ymin) {
          size_t ixbin = (xvalue - m_xmin) / m_xbinSize;
          size_t iybin = (yvalue - m_ymin) / m_ybinSize;
          std::get<2>(Base::m_plotData[iybin * m_nxbins + ixbin]) += weight;
        }
      }

      // this one can ( and in general should ) be overridden - the implementation should use fillWithValue
      bool fill() override {
        auto tag = PlotBase::getTag<0>();
        for (auto iov : tag.iovs) {
          std::shared_ptr<PayloadType> payload = Base::fetchPayload(std::get<1>(iov));
          if (payload.get()) {
            std::tuple<float, float> value = getFromPayload(*payload);
            fillWithValue(std::get<0>(value), std::get<1>(value));
          }
        }
        return true;
      }

      // implement this one if you use the default fill implementation, otherwise ignore it...
      virtual std::tuple<float, float> getFromPayload(PayloadType& payload) {
        float x = 0;
        float y = 0;
        return std::make_tuple(x, y);
      }

    private:
      size_t m_nxbins;
      float m_xbinSize = 0;
      float m_xmin;
      float m_xmax;
      float m_ybinSize = 0;
      size_t m_nybins;
      float m_ymin;
      float m_ymax;
    };

    //
    template <typename PayloadType, IOVMultiplicity IOV_M = UNSPECIFIED_IOV, int NTAGS = 0>
    class PlotImage : public PlotImpl<IOV_M, NTAGS> {
    public:
      typedef PlotImpl<IOV_M, NTAGS> Base;
      explicit PlotImage(const std::string& title) : Base("Image", title) {
        std::string payloadTypeName = cond::demangledName(typeid(PayloadType));
        Base::m_plotAnnotations.m[PlotAnnotations::PAYLOAD_TYPE_K] = payloadTypeName;
        m_imageFileName = edm::createGlobalIdentifier() + ".png";
      }

      std::string serializeData() override { return serialize(Base::m_plotAnnotations, m_imageFileName); }

      std::shared_ptr<PayloadType> fetchPayload(const cond::Hash& payloadHash) {
        return PlotBase::fetchPayload<PayloadType>(payloadHash);
      }

    protected:
      std::string m_imageFileName;
    };

  }  // namespace payloadInspector

}  // namespace cond

#endif
