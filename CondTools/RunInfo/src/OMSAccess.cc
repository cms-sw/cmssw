#include "CondTools/RunInfo/interface/OMSAccess.h"
#include "CondCore/CondDB/interface/WebUtils.h"
#include "CondCore/CondDB/interface/Exception.h"

namespace cond {

  OMSServiceResultRef::OMSServiceResultRef(const boost::property_tree::ptree* row) : m_row(row) {}

  bool OMSServiceResultRef::empty() { return m_row == nullptr; }

  std::string OMSServiceResultRef::getAttribute(const std::string& attributeName) {
    return m_row->get<std::string>(attributeName);
  }

  OMSServiceResultIterator::OMSServiceResultIterator(boost::property_tree::ptree::const_iterator iter) : m_iter(iter) {}

  OMSServiceResultRef OMSServiceResultIterator::operator*() {
    auto& attributeList = m_iter->second.get_child("attributes");
    return OMSServiceResultRef(&attributeList);
  }

  OMSServiceResultIterator& OMSServiceResultIterator::operator++() {
    m_iter++;
    return *this;
  }

  bool OMSServiceResultIterator::operator==(const OMSServiceResultIterator& rhs) { return m_iter == rhs.m_iter; }
  bool OMSServiceResultIterator::operator!=(const OMSServiceResultIterator& rhs) { return m_iter != rhs.m_iter; }

  OMSServiceResult::OMSServiceResult() {}

  OMSServiceResultIterator OMSServiceResult::begin() const { return OMSServiceResultIterator(m_data->begin()); }

  OMSServiceResultIterator OMSServiceResult::end() const { return OMSServiceResultIterator(m_data->end()); }

  OMSServiceResultRef OMSServiceResult::front() const {
    auto& attributeList = m_data->front().second.get_child("attributes");
    return OMSServiceResultRef(&attributeList);
  }

  OMSServiceResultRef OMSServiceResult::back() const {
    auto& attributeList = m_data->back().second.get_child("attributes");
    return OMSServiceResultRef(&attributeList);
  }

  size_t OMSServiceResult::parseData(const std::string& data) {
    m_data = nullptr;
    std::stringstream sout;
    sout << data;
    try {
      boost::property_tree::read_json(sout, m_root);
    } catch (boost::property_tree::json_parser_error const& ex) {
      throw cond::Exception(ex.what(), "OMSServiceResult::parseData");
    }
    if (!m_root.empty()) {
      m_data = &m_root.get_child("data");
    }
    return m_root.size();
  }

  size_t OMSServiceResult::size() const {
    size_t ret = 0;
    if (m_data) {
      ret = m_data->size();
    }
    return ret;
  }

  bool OMSServiceResult::empty() const { return size() == 0; }

  void OMSServiceQuery::addVar(const std::string& varName) {
    std::stringstream varList;
    if (m_varList.empty()) {
      varList << "&fields=";
    } else {
      varList << m_varList << ",";
    }
    varList << varName;
    m_varList = varList.str();
  }

  OMSServiceQuery::OMSServiceQuery(const std::string& baseUrl, const std::string& function) {
    m_url = baseUrl + "/" + function;
  }

  OMSServiceQuery& OMSServiceQuery::addOutputVar(const std::string& varName) {
    addVar(varName);
    return *this;
  }
  OMSServiceQuery& OMSServiceQuery::addOutputVars(const std::initializer_list<const char*>& varNames) {
    for (auto v : varNames)
      addVar(v);
    return *this;
  }

  OMSServiceQuery& OMSServiceQuery::limit(int value) {
    std::stringstream pageLimit;
    if (m_filter.empty()) {
      pageLimit << "?";
    } else {
      pageLimit << "&";
    }
    pageLimit << "page[limit]=" << value;
    m_limit = pageLimit.str();
    return *this;
  }

  bool OMSServiceQuery::execute() {
    bool ret = false;
    std::string out;
    m_status = cond::httpGet(url(), out);
    if (m_status == 200 || m_status == 201) {
      m_result = std::make_unique<OMSServiceResult>();
      m_result->parseData(out);
      ret = true;
    }
    return ret;
  }

  unsigned long OMSServiceQuery::status() { return m_status; }

  OMSServiceResult& OMSServiceQuery::result() { return *m_result; }

  std::string OMSServiceQuery::url() { return m_url + m_filter + m_limit + m_varList; }

  OMSService::OMSService() : m_baseUrl() {}

  void OMSService::connect(const std::string& baseUrl) { m_baseUrl = baseUrl; }
  std::unique_ptr<OMSServiceQuery> OMSService::query(const std::string& function) const {
    return std::make_unique<OMSServiceQuery>(m_baseUrl, function);
  }
}  // namespace cond
