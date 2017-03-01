#ifndef FWCore_Catalog_FileLocator_h
#define FWCore_Catalog_FileLocator_h

#include <string>
#include <list>
#include <map>
#include <utility>
#include <regex>
#include <xercesc/dom/DOM.hpp>

namespace edm {

  class FileLocator {

  public:
    explicit FileLocator(std::string const& catUrl, bool fallback);
    ~FileLocator();

    std::string pfn(std::string const& ilfn) const;
    std::string lfn(std::string const& ipfn) const;

  private:
    /** For the time being the only allowed configuration item is a
     *  prefix to be added to the GUID/LFN.
     */
    static int s_numberOfInstances;

    struct Rule {
      std::regex pathMatch;
      std::regex destinationMatch;
      std::string result;
      std::string chain;
    };

    typedef std::vector<Rule> Rules;
    typedef std::map<std::string, Rules> ProtocolRules;

    void init(std::string const& catUrl, bool fallback);

    void parseRule(xercesc::DOMNode* ruleNode,
                   ProtocolRules& rules);

    std::string applyRules(ProtocolRules const& protocolRules,
                           std::string const& protocol,
                           std::string const& destination,
                           bool direct,
                           std::string name) const;

    std::string convert(std::string const& input, ProtocolRules const& rules, bool direct) const;

    /** Direct rules are used to do the mapping from LFN to PFN.*/
    ProtocolRules m_directRules;
    /** Inverse rules are used to do the mapping from PFN to LFN*/
    ProtocolRules m_inverseRules;

    std::string                 m_fileType;
    std::string                 m_filename;
    std::vector<std::string>    m_protocols;
    std::string                 m_destination;
  };
}

#endif //  FWCore_Catalog_FileLocator_h
