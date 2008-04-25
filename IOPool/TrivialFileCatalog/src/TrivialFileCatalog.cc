/* 
   Concrete implementation of a FileCatalog.
   Author: Giulio.Eulisse@cern.ch
 */

#include <set>
#include <string>
#include <stdexcept>
#ifndef POOL_TRIVIALFILECATALOG_H
#include "IOPool/TrivialFileCatalog/interface/TrivialFileCatalog.h"
#endif
#include "POOLCore/PoolMessageStream.h"
#include "FileCatalog/FCException.h"


#include <xercesc/dom/DOM.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/util/PlatformUtils.hpp>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <SealBase/StringList.h>
#include <SealBase/StringOps.h>
#include <SealBase/DebugAids.h>
#include <SealBase/Regexp.h>

using namespace xercesc;

int pool::TrivialFileCatalog::s_numberOfInstances = 0;

inline std::string _toString(const XMLCh *toTranscode)
{
    std::string tmp(XMLString::transcode(toTranscode));
    return tmp;
}

inline XMLCh*  _toDOMS(std::string temp){
    XMLCh* buff = XMLString::transcode(temp.c_str());    
    return  buff;
}

pool::TrivialFileCatalog::TrivialFileCatalog ()
    : m_connectionStatus (false),
      m_fileType ("ROOT_All"),
      m_destination ("any")
{  
    PoolMessageStream trivialLog("TrivialFileCatalog", seal::Msg::Nil);
    try { 
	trivialLog <<seal::Msg::Info << "Xerces-c initialization Number "
	  << s_numberOfInstances <<seal::endmsg;
	if (s_numberOfInstances==0) 
	    XMLPlatformUtils::Initialize();  
    }
    catch (const XMLException& e) {
	trivialLog <<seal::Msg::Fatal << "Xerces-c error in initialization \n"
	      << "Exception message is:  \n"
	      << _toString(e.getMessage()) << seal::endmsg;
        throw(std::runtime_error("Standard pool exception : Fatal Error on pool::TrivialFileCatalog"));
    }
    ++s_numberOfInstances;
    
}

pool::TrivialFileCatalog::~TrivialFileCatalog ()
{
}

void
pool::TrivialFileCatalog::parseRule (DOMNode *ruleNode, bool direct) 
{
    if (!ruleNode)
    {
	throw FCTransactionException
	    ("TrivialFileCatalog::connect",
	     ":Malformed trivial catalog"); 		
    }
	    
    DOMElement* ruleElement = static_cast<DOMElement *>(ruleNode);	    

    if (! ruleElement)
    {
	throw FCTransactionException
	    ("TrivialFileCatalog::connect",
	     ":Malformed trivial catalog"); 		
    }
	    
    this->addRule (_toString (ruleElement->getAttribute (_toDOMS ("protocol"))),
                   _toString (ruleElement->getAttribute (_toDOMS ("destination-match"))),
                   _toString (ruleElement->getAttribute (_toDOMS ("path-match"))),
                   _toString (ruleElement->getAttribute (_toDOMS ("result"))),
                   _toString (ruleElement->getAttribute (_toDOMS ("chain"))), direct);
}

void
pool::TrivialFileCatalog::addRule (const std::string &protocol,
                                   const std::string &destinationMatchRegexp,
                                   const std::string &pathMatchRegexp,
                                   const std::string &result,
                                   const std::string &chain, 
                                   bool direct, bool back)
{
    std::string _destMatchRegexp = ".*";
    if (!destinationMatchRegexp.empty ()) 
    _destMatchRegexp = destinationMatchRegexp; 

    Rule rule;
    rule.pathMatch.setPattern (pathMatchRegexp);
    rule.pathMatch.compile ();
    rule.destinationMatch.setPattern (_destMatchRegexp);
    rule.destinationMatch.compile ();
    rule.result = result;
    rule.chain = chain;
    if (direct) {
        if (back) {
            m_directRules[protocol].push_back (rule);
        } else {
            m_directRules[protocol].push_front (rule);
        }
    } else {
        if (back) {
            m_inverseRules[protocol].push_back (rule);
        } else {
            m_inverseRules[protocol].push_front (rule);
        }
    }
}

void
pool::TrivialFileCatalog::connect ()
{
    try
    {
	PoolMessageStream trivialLog("TrivialFileCatalog", seal::Msg::Nil);
  	trivialLog << seal::Msg::Info << "Connecting to the catalog "
		   << m_url << seal::endmsg;

	if (m_url.find ("file:") != std::string::npos)
	{
	    m_url = m_url.erase (0, 
				 m_url.find (":") + 1);	
	}	
	else
	{
	    throw FCTransactionException
		("TrivialFileCatalog::connect",
		 ": Malformed url for file catalog configuration"); 
	}

	seal::StringList tokens = seal::StringOps::split (m_url, "?"); 
	m_filename = tokens[0];

	if (tokens.size () == 2)
	{
	    std::string options = tokens[1];
	    seal::StringList optionTokens = seal::StringOps::split (options, "&");

	    for (seal::StringList::iterator option = optionTokens.begin ();
		 option != optionTokens.end ();
		 option++)
	    {
		seal::StringList argTokens = seal::StringOps::split (*option, "=") ;
		if (argTokens.size () != 2)
		{
		    throw FCTransactionException
			("TrivialFileCatalog::connect",
			 ": Malformed url for file catalog configuration"); 
		}
		
		std::string key = argTokens[0];
		std::string value = argTokens[1];
		
		if (key == "protocol")
		{
		    m_protocols = seal::StringOps::split (value, ",");
		}
		else if (key == "destination")
		{
		    m_destination = value;
		}
	    }
	}
	
	if (m_protocols.empty ())
	    throw FCTransactionException
		("TrivialFileCatalog::connect",
		 ": protocol was not supplied in the contact string"); 
		
	std::ifstream configFile;
	configFile.open (m_filename.c_str ());
	
	
	trivialLog << seal::Msg::Info
		   << "Using catalog configuration " 
		   << m_filename << seal::endmsg;
	
	if (!configFile.good () || !configFile.is_open ())
	{
	    m_transactionsta = 0;
	    throw FCTransactionException
		("TrivialFileCatalog::connect",
		 ": Unable to open trivial file catalog " + m_filename); 
	}
	
	configFile.close ();
	
	XercesDOMParser* parser = new XercesDOMParser;     
	parser->setValidationScheme(XercesDOMParser::Val_Auto);
	parser->setDoNamespaces(false);
	parser->parse(m_filename.c_str());	
	DOMDocument* doc = parser->getDocument();
	ASSERT (doc);
	
	/* trivialFileCatalog matches the following xml schema
	   FIXME: write a proper DTD
	    <storage-mapping>
	    <lfn-to-pfn protocol="direct" destination-match=".*" 
	    path-match="lfn/guid match regular expression"
	    result="/castor/cern.ch/cms/$1"/>
	    <pfn-to-lfn protocol="srm" 
	    path-match="lfn/guid match regular expression"
	    result="$1"/>
	    </storage-mapping>
	 */

	/*first of all do the lfn-to-pfn bit*/
	{
	    DOMNodeList *rules =doc->getElementsByTagName(_toDOMS("lfn-to-pfn"));
	    unsigned int ruleTagsNum  = 
		rules->getLength();
	
	    // FIXME: we should probably use a DTD for checking validity 

	    for (unsigned int i=0; i<ruleTagsNum; i++) {
		DOMNode* ruleNode =	rules->item(i);
		parseRule (ruleNode, true);
	    }
	}
	/*Then we handle the pfn-to-lfn bit*/
	{
	    DOMNodeList *rules =doc->getElementsByTagName(_toDOMS("pfn-to-lfn"));
	    unsigned int ruleTagsNum  = 
		rules->getLength();
	
	    for (unsigned int i=0; i<ruleTagsNum; i++){
		DOMNode* ruleNode =	rules->item(i);
		parseRule (ruleNode, false);
	    }	    
	}
	
	m_transactionsta = 1;
    }
    catch(std::exception& er)
    {
	m_transactionsta = 0;	
	throw FCconnectionException("TrivialFileCatalog::connect",er.what());
    }
}

void
pool::TrivialFileCatalog::disconnect () const
{
	m_transactionsta = 0;	    
}

void
pool::TrivialFileCatalog::start () const
{
}

void
pool::TrivialFileCatalog::commit (FileCatalog::CommitMode /*cm=FileCatalog::REFRESH*/) const
{
}

void
pool::TrivialFileCatalog::rollback () const
{
    throw FCTransactionException
	("TrivialFileCatalog::rollback",
	 "Trivial catalogs cannot rollback");
}
  
void
pool::TrivialFileCatalog::registerPFN (const std::string& /*pfn*/, 
				       const std::string& /*filetype*/,
				       FileCatalog::FileID& /*fid*/) const
{
    throw FCTransactionException
	("TrivialFileCatalog::registerPFN",
	 "It does not make sense to register PFN for TriviaFileCatalogs");    
}   

void
pool::TrivialFileCatalog::registerLFN (const std::string& /*pfn*/, 
				       const std::string& /*lfn*/) const
{
    throw FCTransactionException
	("TrivialFileCatalog::registerLFN",
	 "It does not make sense to register LFN for TriviaFileCatalogs");    
}

void
pool::TrivialFileCatalog::addReplicaPFN (const std::string& /*pfn*/, 
					 const std::string& /*rpf*/) const
{
    throw FCTransactionException
	("TrivialFileCatalog::registerLFN",
	 "It does not make sense to register PFN for TriviaFileCatalogs");    
}

void
pool::TrivialFileCatalog::addPFNtoGuid (const FileCatalog::FileID& /*guid*/, 
					const std::string& /*pf*/, 
					const std::string& /*filetype*/) const
{
    throw FCTransactionException
	("TrivialFileCatalog::addPFNtoGuid",
	 "It does not make sense to register replicas for TriviaFileCatalogs");    
}

void
pool::TrivialFileCatalog::renamePFN (const std::string& /*pfn*/, 
				     const std::string& /*newpfn*/) const
{
    throw FCTransactionException
	("TrivialFileCatalog::renamePFN",
	 "It does not make sense to rename PFNs for TrivialFileCatalogs");    
}

void
pool::TrivialFileCatalog::lookupFileByPFN (const std::string & pfn, 
					   FileCatalog::FileID & fid, 
					   std::string& filetype) const
{
    filetype = m_fileType;    
    fid = "";
    std::string tmpPfn = pfn;
    
    for (seal::StringList::const_iterator protocol = m_protocols.begin ();
	 protocol != m_protocols.end ();
	 protocol++)
    {
	fid = applyRules (m_inverseRules, *protocol, m_destination, false, tmpPfn);
	if (! fid.empty ())
	{
	    return;	    
	}	
    }
}

void
pool::TrivialFileCatalog::lookupFileByLFN (const std::string& lfn, 
					   FileCatalog::FileID& fid) const
{
    // return NULL id if the catalog is not connected.
    if (m_transactionsta == 0) {
	fid = FileCatalog::FileID();
	return;
    }
	
    // GUID (FileID) and lfn are the same under TrivialFileCatalog
    fid = lfn;    
}

std::string
replaceWithRegexp (const seal::RegexpMatch matches, 
		   const std::string inputString,
		   const std::string outputFormat)
{
    //std::cerr << "InputString:" << inputString << std::endl;
    
    char buffer[8];
    std::string result = outputFormat;
        
    for (int i = 1;
	 i < matches.numMatches ();
	 i++)
    {
	// If this is not true, man, we are in trouble...
	ASSERT (i < 1000000);
	sprintf (buffer, "%i", i);
	std::string variableRegexp = std::string ("[$]") + buffer;
	std::string matchResult = matches.matchString (inputString, i);
	
	seal::Regexp sustitutionToken (variableRegexp);
	
	//std::cerr << "Current match: " << matchResult << std::endl;
	
	result = seal::StringOps::replace (result, 
					   sustitutionToken, 
					   matchResult);
    }
    return result;    
}


std::string 
pool::TrivialFileCatalog::applyRules (const ProtocolRules& protocolRules,
				      const std::string & protocol,
				      const std::string & destination,
				      bool direct,
				      std::string name) const
{
    //std::cerr << "Calling apply rules with protocol: " << protocol << "\n destination: " << destination << "\n " << " on name " << name << std::endl;
    
    const ProtocolRules::const_iterator rulesIterator = protocolRules.find (protocol);
    if (rulesIterator == protocolRules.end ())
	return "";
    
    const Rules &rules=(*(rulesIterator)).second;
    
    /* Look up for a matching rule*/
    for (Rules::const_iterator i = rules.begin ();
	 i != rules.end ();
	 i++)
    {
	if (! i->destinationMatch.exactMatch (destination))
	    continue;
	
	if (! i->pathMatch.exactMatch (name))
	    continue;
	
	//std::cerr << "Rule " << i->pathMatch.pattern () << "matched! " << std::endl;	
	
	std::string chain = i->chain;
	if ((direct==true) && (chain != ""))
	{
	    name = 
		applyRules (protocolRules, chain, destination, direct, name);		
	}
	    
	seal::RegexpMatch matches;
	i->pathMatch.match (name, 0, 0, &matches);
	
	name = replaceWithRegexp (matches, 
				  name,
				  i->result); 
	    
	if ((direct == false) && (chain !=""))
	{	
	    name = 
		applyRules (protocolRules, chain, destination, direct, name);		
	}
	
	return name;
    }
    return "";
}


void
pool::TrivialFileCatalog::lookupBestPFN (const FileCatalog::FileID& fid, 
					 const FileCatalog::FileOpenMode& /*omode*/,
					 const FileCatalog::FileAccessPattern& /*amode*/,
					 std::string& pfn,
					 std::string& filetype) const
{
    if (m_transactionsta == 0)
	throw FCconnectionException("TrivialFileCatalog::lookupBestPFN",
				    "Catalog not connected");
    filetype = m_fileType;    
    
    pfn = "";
    std::string lfn = fid;
    
    for (seal::StringList::const_iterator protocol = m_protocols.begin ();
	 protocol != m_protocols.end ();
	 protocol++)
    {
	pfn = applyRules (m_directRules, 
			  *protocol, 
			  m_destination, 
			  true, 
			  lfn);
	if (! pfn.empty ())
	{
	    return;
	}
    }
} 

void
pool::TrivialFileCatalog::insertPFN (PFNEntry& /*pentry*/) const
{
    throw FCTransactionException
	("TrivialFileCatalog::insertPFN",
	 "It does not make sense to insert PFNs for TrivialFileCatalogs");    
}

void
pool::TrivialFileCatalog::insertLFN (LFNEntry& /*lentry*/) const
{
    throw FCTransactionException
	("TrivialFileCatalog::insertLFN",
	 "It does not make sense to insert LFNs for TrivialFileCatalogs");    
}

void
pool::TrivialFileCatalog::deletePFN (const std::string& /*pfn*/) const
{
    throw FCTransactionException
	("TrivialFileCatalog::deletePFN",
	 "It does not make sense to delete PFNs for TrivialFileCatalogs");    
}

void
pool::TrivialFileCatalog::deleteLFN (const std::string& /*lfn*/) const
{
    throw FCTransactionException
	("TrivialFileCatalog::deleteLFN",
	 "It does not make sense to delete LFNs for TrivialFileCatalogs");    
}

void
pool::TrivialFileCatalog::deleteEntry (const FileCatalog::FileID& /*guid*/) const
{
    throw FCTransactionException
	("TrivialFileCatalog::deleteEntry",
	 "It does not make sense to delete GUIDs for TrivialFileCatalogs");    
}

bool
pool::TrivialFileCatalog::isReadOnly () const
{
    return true;    
}
 
bool
pool::TrivialFileCatalog::retrievePFN (const std::string& query, 
				       FCBuf<PFNEntry>& buf, 
				       const size_t& /*start*/)
{
    if (m_transactionsta == 0)
	throw FCconnectionException("TrivialFileCatalog::lookupBestPFN",
				    "Catalog not connected");
    // The only query supported is lfn='something' or pfn='something'
    // No spaces allowed in something.
    seal::Regexp grammar ("(lfname|guid)='(.*)'");
    seal::RegexpMatch grammarMatches;
    
    grammar.match (query, 0, 0, &grammarMatches);
    
    if (grammarMatches.numMatches () != 3)
    {
	throw FCTransactionException
	    ("TrivialFileCatalog::retrievePFN",
	     "malformed query. the only supported one is lfname='something'"
	     " or guid='something'");
    }
    
    std::string lfn = grammarMatches.matchString (query, 2);
    
    for (seal::StringList::iterator protocol = m_protocols.begin ();
	 protocol != m_protocols.end ();
	 protocol++)
    {
	std::string pfn = applyRules (m_directRules, *protocol, m_destination, true, lfn);
	if (! pfn.empty ())
	{
	    buf.push_back (PFNEntry(pfn, 
				    lfn, 
				    m_fileType));    
	    return true;    
	}	
    }


    
    buf.push_back (PFNEntry(lfn, 
			    lfn, 
			    m_fileType));    
    
    return false;    
}

bool
pool::TrivialFileCatalog::retrieveLFN (const std::string& query, 
				       FCBuf<LFNEntry>& buf, 
				       const size_t& /*start*/)
{    
    if (m_transactionsta == 0)
	throw FCconnectionException("TrivialFileCatalog::lookupBestPFN",
				    "Catalog not connected");
    // The only query supported is lfn='something' or pfn='something'
    // No spaces allowed in something.

    seal::Regexp grammar ("(pfname|guid)='(.*)'");
    seal::RegexpMatch grammarMatches;
    
    grammar.match (query, 0, 0, &grammarMatches);
    
    if (grammarMatches.numMatches () != 3)
    {
	throw FCTransactionException
	    ("TrivialFileCatalog::retrieveLFN",
	     "malformed query. the only supported one is pfname='something'"
	     " or guid='something'");
    }

    std::string selector = grammarMatches.matchString (query, 1);
    std::string pfn = grammarMatches.matchString (query, 2);


    if (selector == "guid")
    {
	buf.push_back (LFNEntry (pfn,
				 pfn));
	return true;	
    }
    

    for (seal::StringList::iterator protocol = m_protocols.begin ();
	 protocol != m_protocols.end ();
	 protocol++)
    {
	std::string lfn = applyRules (m_inverseRules, *protocol, m_destination, false, pfn);
	//	std::cerr << "LFN: " << lfn << std::endl;
	
	if (! lfn.empty ())
	{
	    buf.push_back (LFNEntry(lfn, 
				    lfn));    
	    return true;    
	}	
    }


    
    buf.push_back (LFNEntry(pfn, 
			    pfn));    
    
    return false;    
}

bool
pool::TrivialFileCatalog::retrieveGuid (const std::string& query, 
					FCBuf<FileCatalog::FileID>& buf, 
					const size_t& start)
{   
    typedef FCBuf<LFNEntry> Buffer;
    
    Buffer tmpBuf (10);
    bool result = retrieveLFN (query, tmpBuf, start);
    std::vector<LFNEntry> tmpVect = tmpBuf.getData ();
    
    for (std::vector<LFNEntry>::iterator i = tmpVect.begin ();
	 i != tmpVect.end ();
	 i++)
    {
	buf.push_back (i->lfname ());	
    }    

    return result;    
}
