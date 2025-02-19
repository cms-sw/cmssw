#ifndef ConfigurationDatabaseStandardXMLParser_hh_included
#define ConfigurationDatabaseStandardXMLParser_hh_included 1

#include "xercesc/sax2/SAX2XMLReader.hpp"
#include "CaloOnlineTools/HcalOnlineDb/interface/ConfigurationDatabaseException.hh"
#include <string>
#include <vector>
#include <map>
#include <list>

/** \brief SAX-based XML parser for "CFGBRICK"-style configuration information.
 */

class ConfigurationDatabaseStandardXMLParser {
public:
  ConfigurationDatabaseStandardXMLParser();
  void parse(const std::string& xmlDocument, std::map<std::string,std::string>& parameters, std::vector<std::string>& items, std::string& encoding) throw (hcal::exception::ConfigurationDatabaseException);
  struct Item {
    std::map<std::string,std::string> parameters;
    std::vector<std::string> items;
    std::string encoding;
    std::vector<unsigned int> convert() const;
  };
  void parseMultiple(const std::string& xmlDocument, std::list<Item>& items) throw (hcal::exception::ConfigurationDatabaseException);
private:
  xercesc::SAX2XMLReader* m_parser;
};


#endif // ConfigurationDatabaseStandardXMLParser_hh_included
