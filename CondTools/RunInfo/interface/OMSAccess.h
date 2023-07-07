#ifndef CondTools_RunInfo_OMSAccess_h
#define CondTools_RunInfo_OMSAccess_h

#include <string>
#include <stdexcept>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <memory>

namespace cond {

  // implementation details for data extraction from json/ data conversion to build query urls
  namespace impl {

    static constexpr const char* const OMS_TIME_FMT = "%Y-%m-%dT%H:%M:%SZ";

    template <typename T, T fun(const std::string&)>
    inline T from_string_impl(const std::string& attributeValue, T zero) {
      T ret = zero;
      if (not attributeValue.empty() && attributeValue != "null") {
        ret = fun(attributeValue);
      }
      return ret;
    }

    template <typename T>
    inline T from_string(const std::string& attributeValue) {
      throw std::invalid_argument("");
    }

    template <>
    inline std::string from_string(const std::string& attributeValue) {
      return std::string(attributeValue);
    }

    inline float s_to_f(const std::string& val) { return std::stof(val); }
    template <>
    inline float from_string(const std::string& attributeValue) {
      return from_string_impl<float, &s_to_f>(attributeValue, 0.);
    }

    inline int s_to_i(const std::string& val) { return std::stoi(val); }
    template <>
    inline int from_string(const std::string& attributeValue) {
      return from_string_impl<int, &s_to_i>(attributeValue, 0);
    }

    inline unsigned long s_to_ul(const std::string& val) { return std::stoul(val); }
    template <>
    inline unsigned short from_string(const std::string& attributeValue) {
      unsigned long int_val = from_string_impl<unsigned long, &s_to_ul>(attributeValue, 0);
      return (unsigned short)int_val;
    }
    inline unsigned long long s_to_ull(const std::string& val) { return std::stoull(val); }
    template <>
    inline unsigned long long from_string(const std::string& attributeValue) {
      unsigned long long int_val = from_string_impl<unsigned long long, &s_to_ull>(attributeValue, 0);
      return int_val;
    }

    inline boost::posix_time::ptime s_to_time(const std::string& val) {
      boost::posix_time::time_input_facet* facet = new boost::posix_time::time_input_facet(OMS_TIME_FMT);
      std::stringstream ss;
      ss.imbue(std::locale(std::locale(), facet));
      ss << val;
      boost::posix_time::ptime time;
      ss >> time;
      return time;
    }
    template <>
    inline boost::posix_time::ptime from_string(const std::string& attributeValue) {
      return from_string_impl<boost::posix_time::ptime, &s_to_time>(attributeValue, boost::posix_time::ptime());
    }

    template <typename V>
    inline std::string to_string(const V& value) {
      return std::to_string(value);
    }
    template <typename V>
    inline std::string to_string(const V* value) {
      return std::to_string(*value);
    }
    template <>
    inline std::string to_string(const std::string& value) {
      return std::string(value);
    }
    template <>
    inline std::string to_string(const char* value) {
      return std::string(value);
    }
    template <>
    inline std::string to_string(const boost::posix_time::ptime& value) {
      boost::posix_time::time_facet* facet = new boost::posix_time::time_facet();
      facet->format(OMS_TIME_FMT);
      std::stringstream stream;
      stream.imbue(std::locale(std::locale::classic(), facet));
      stream << value;
      return stream.str();
    }

  }  // namespace impl

  // reference of a result set row. it does not own/hold data.
  class OMSServiceResultRef {
  public:
    OMSServiceResultRef() = delete;
    OMSServiceResultRef(const boost::property_tree::ptree* row);

    // return true if no attribute is available
    bool empty();
    // typed getter for single param
    template <typename T>
    inline T get(const std::string& attributeName) {
      return impl::from_string<T>(getAttribute(attributeName));
    }
    // getter for arrays
    template <typename primitive>
    std::vector<primitive> getArray(const std::string& attributeName) {
      std::vector<primitive> ret;
      for (auto& item : m_row->get_child(attributeName)) {
        ret.push_back(item.second.get_value<primitive>());
      }
      return ret;
    }

  private:
    std::string getAttribute(const std::string& attributeName);
    const boost::property_tree::ptree* m_row = nullptr;
  };

  // iterator object for result
  class OMSServiceResultIterator {
  public:
    OMSServiceResultIterator() = delete;
    OMSServiceResultIterator(boost::property_tree::ptree::const_iterator iter);

    OMSServiceResultRef operator*();
    OMSServiceResultIterator& operator++();

    bool operator==(const OMSServiceResultIterator& rhs);
    bool operator!=(const OMSServiceResultIterator& rhs);

  private:
    boost::property_tree::ptree::const_iterator m_iter;
  };

  // container wrapping the query result, based on boost property tree
  class OMSServiceResult {
  public:
    OMSServiceResult();
    // basic iterators, to enable the C++11 for loop semantics
    OMSServiceResultIterator begin() const;
    OMSServiceResultIterator end() const;

    OMSServiceResultRef front() const;
    OMSServiceResultRef back() const;

    // parse json returned from curl, filling the property tree
    size_t parseData(const std::string& data);

    // returns the number of top level elements of the tree ( result set "rows" )
    size_t size() const;

    // returns size()==0
    bool empty() const;

  private:
    boost::property_tree::ptree m_root;
    boost::property_tree::ptree* m_data;
  };

  // Query object
  class OMSServiceQuery {
  public:
    // comparison operator label, used in query urls
    static constexpr const char* const NEQ = "NEQ";
    static constexpr const char* const EQ = "EQ";
    static constexpr const char* const LT = "LT";
    static constexpr const char* const LE = "LE";
    static constexpr const char* const GT = "GT";
    static constexpr const char* const GE = "GE";
    static constexpr const char* const SNULL = "null";

  public:
    OMSServiceQuery() = delete;
    OMSServiceQuery(const std::string& baseUrl, const std::string& function);

    // functions to restring query output to specific variables
    OMSServiceQuery& addOutputVar(const std::string& varName);
    OMSServiceQuery& addOutputVars(const std::initializer_list<const char*>& varNames);

    // generic query filter
    template <typename T>
    inline OMSServiceQuery& filter(const char* cmp, const std::string& varName, const T& value) {
      std::stringstream filter;
      if (m_filter.empty()) {
        filter << "?";
        if (!m_limit.empty()) {
          m_limit.front() = '&';
        }
      } else {
        filter << m_filter << "&";
      }
      filter << "filter[" << varName << "][" << cmp << "]=" << impl::to_string(value);
      m_filter = filter.str();
      return *this;
    }
    // filters with specific comparison operators
    template <typename T>
    inline OMSServiceQuery& filterEQ(const std::string& varName, const T& value) {
      return filter<T>(EQ, varName, value);
    }
    template <typename T>
    inline OMSServiceQuery& filterNEQ(const std::string& varName, const T& value) {
      return filter<T>(NEQ, varName, value);
    }
    template <typename T>
    inline OMSServiceQuery& filterGT(const std::string& varName, const T& value) {
      return filter<T>(GT, varName, value);
    }
    template <typename T>
    inline OMSServiceQuery& filterGE(const std::string& varName, const T& value) {
      return filter<T>(GE, varName, value);
    }
    template <typename T>
    inline OMSServiceQuery& filterLT(const std::string& varName, const T& value) {
      return filter<T>(LT, varName, value);
    }
    template <typename T>
    inline OMSServiceQuery& filterLE(const std::string& varName, const T& value) {
      return filter<T>(LE, varName, value);
    }
    // not null filter
    inline OMSServiceQuery& filterNotNull(const std::string& varName) { return filterNEQ(varName, SNULL); }

    // limit for the page size, when unspecified OMS's default limit is 100
    OMSServiceQuery& limit(int value);

    // triggers the execution of the query ( calling curl functions )
    bool execute();

    // return code from curl
    unsigned long status();

    // result from the query. memory allocated for data is owned by the query object itself
    OMSServiceResult& result();

    // the url constructed and used for the query
    std::string url();

  private:
    void addVar(const std::string& varName);

  private:
    std::string m_url;
    std::string m_filter;
    std::string m_limit;
    std::string m_varList;
    std::unique_ptr<OMSServiceResult> m_result;
    unsigned long m_status = 0;
  };

  // provides query access to OMS Web services
  class OMSService {
  public:
    OMSService();

    void connect(const std::string& baseUrl);
    std::unique_ptr<OMSServiceQuery> query(const std::string& function) const;

  private:
    std::string m_baseUrl;
  };
}  // namespace cond

#endif
