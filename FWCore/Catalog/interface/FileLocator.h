#ifndef FWCore_Catalog_FileLocator_h
#define FWCore_Catalog_FileLocator_h

#include <string>
#include <list>
#include <map>
#include <utility>
#include "classlib/utils/Regexp.h"
#include <xercesc/dom/DOM.hpp>

namespace edm {

  class FileLocator {

  public:
    FileLocator();
    ~FileLocator();

    std::string pfn(std::string const & ilfn) const;
    std::string lfn(std::string const & ipfn) const;

  private:

private:
    /** For the time being the only allowed configuration item is a
     *  prefix to be added to the GUID/LFN.
     */ 
    static int		s_numberOfInstances;    

    
    
    typedef struct {
	lat::Regexp pathMatch;
	lat::Regexp destinationMatch;	
	std::string result;
	std::string chain;
    } Rule;

    typedef std::list <Rule> Rules;
    typedef std::map <std::string, Rules> ProtocolRules;

    void init();

    void parseRule (xercesc::DOMNode *ruleNode, 
		    ProtocolRules &rules);
    
    std::string applyRules (const ProtocolRules& protocolRules,
			    const std::string & protocol,
			    const std::string & destination,
			    bool direct,
			    std::string name) const;
    

            
    /** Direct rules are used to do the mapping from LFN to PFN.*/
    ProtocolRules	 	m_directRules;
    /** Inverse rules are used to do the mapping from PFN to LFN*/
    ProtocolRules		m_inverseRules;
    
    std::string 		m_fileType;
    std::string			m_filename;
    lat::StringList		m_protocols;
    std::string			m_destination; 
  };

}


#endif //  FWCore_Catalog_FileLocator_h


