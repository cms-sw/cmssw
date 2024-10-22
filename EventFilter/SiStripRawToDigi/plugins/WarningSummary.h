#ifndef EventFilter_SiStripRawToDigi_WarningSummary
#define EventFilter_SiStripRawToDigi_WarningSummary 1

#include <string>
#include <map>

namespace sistrip {
  class WarningSummary {
  public:
    WarningSummary(const std::string& category, const std::string& name, bool debug = false)
        : m_debug(debug), m_category(category), m_name(name) {}

    void add(const std::string& message, const std::string& details = "");
    void printSummary() const;

  private:
    bool m_debug;
    std::string m_category;
    std::string m_name;
    std::map<std::string, std::size_t> m_warnings;
  };
}  // namespace sistrip
#endif  // EventFilter_SiStripRawToDigi_WarningSummary
