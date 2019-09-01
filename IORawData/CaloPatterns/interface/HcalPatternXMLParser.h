#ifndef HcalPatternXMLParser_hh_included
#define HcalPatternXMLParser_hh_included 1

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <cstdint>

class HcalPatternXMLParserImpl;

class HcalPatternXMLParser {
public:
  HcalPatternXMLParser();
  ~HcalPatternXMLParser();
  void parse(const std::string& xmlDocument,
             std::map<std::string, std::string>& parameters,
             std::vector<std::string>& items,
             std::string& encoding);
  void parse(const std::string& xmlDocument,
             std::map<std::string, std::string>& parameters,
             std::vector<uint32_t>& items);

private:
  HcalPatternXMLParserImpl* m_parser;
};

#endif  // HcalPatternXMLParser_hh_included
