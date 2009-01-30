/* 
   Concrete implementation of a FileCatalog.
   Author: Giulio.Eulisse@cern.ch
 */

#include <set>
#include <string>
#include <stdexcept>
#include <cassert>
#ifndef POOL_TRIVIALFILECATALOG_H
#include "TrivialFileCatalog.h"
#endif
#include "CoralBase/MessageStream.h"
#include "FileCatalog/FCException.h"

#include <xercesc/dom/DOM.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/util/PlatformUtils.hpp>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include "classlib/utils/StringList.h"
#include "classlib/utils/StringOps.h"
#include "classlib/utils/Regexp.h"
#include "Reflex/PluginService.h"
using namespace xercesc;
using namespace pool;

PLUGINSVC_FACTORY_WITH_ID( TrivialFileCatalog,std::string("trivialcatalog"),FCImpl*())

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
    coral::MessageStream trivialLog("TrivialFileCatalog");
    try { 
	trivialLog <<coral::Info << "Xerces-c initialization Number "
	  << s_numberOfInstances <<coral::MessageStream::endmsg;
	if (s_numberOfInstances==0) 
	    XMLPlatformUtils::Initialize();  
    }
    catch (const XMLException& e) {
	trivialLog <<coral::Fatal << "Xerces-c error in initialization \n"
	      << "Exception message is:  \n"
	      << _toString(e.getMessage()) <<coral::MessageStream::endmsg;
        throw(std::runtime_error("Standard pool exception : Fatal Error on pool::TrivialFileCatalog"));
    }
    ++s_numberOfInstances;
    
}

pool::TrivialFileCatalog::~TrivialFileCatalog ()
{
}

void
pool::TrivialFileCatalog::parseRule (DOMNode *ruleNode, 
				     ProtocolRules &rules)
{
    if (! ruleNode)
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
	    
    std::string protocol 
	= _toString (ruleElement->getAttribute (_toDOMS ("protocol")));	    
    std::string destinationMatchRegexp
	= _toString (ruleElement->getAttribute (_toDOMS ("destination-match")));

    if (destinationMatchRegexp.empty ())
	destinationMatchRegexp = ".*";

    std::string pathMatchRegexp 
	= _toString (ruleElement->getAttribute (_toDOMS ("path-match")));
    std::string result 
	= _toString (ruleElement->getAttribute (_toDOMS ("result")));
    std::string chain 
	= _toString (ruleElement->getAttribute (_toDOMS ("chain")));
    					    
    Rule rule;
    rule.pathMatch.setPattern (pathMatchRegexp);
    rule.pathMatch.compile ();
    rule.destinationMatch.setPattern (destinationMatchRegexp);
    rule.destinationMatch.compile ();    
    rule.result = result;
    rule.chain = chain;
    rules[protocol].push_back (rule);    
}

void
pool::TrivialFileCatalog::connect ()
{
    try
    {
	coral::MessageStream trivialLog("TrivialFileCatalog");
  	trivialLog << coral::Info << "Connecting to the catalog "
		   << m_url << coral::MessageStream::endmsg;

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

	lat::StringList tokens = lat::StringOps::split (m_url, "?"); 
	m_filename = tokens[0];

	if (tokens.size () == 2)
	{
	    std::string options = tokens[1];
	    lat::StringList optionTokens = lat::StringOps::split (options, "&");

	    for (lat::StringList::iterator option = optionTokens.begin ();
		 option != optionTokens.end ();
		 option++)
	    {
		lat::StringList argTokens = lat::StringOps::split (*option, "=") ;
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
		    m_protocols = lat::StringOps::split (value, ",");
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
	
	
	trivialLog << coral::Info
		   << "Using catalog configuration " 
		   << m_filename << coral::MessageStream::endmsg;
	
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
	assert(doc);
	
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
		parseRule (ruleNode, m_directRules);
	    }
	}
	/*Then we handle the pfn-to-lfn bit*/
	{
	    DOMNodeList *rules =doc->getElementsByTagName(_toDOMS("pfn-to-lfn"));
	    unsigned int ruleTagsNum  = 
		rules->getLength();
	
	    for (unsigned int i=0; i<ruleTagsNum; i++){
		DOMNode* ruleNode =	rules->item(i);
		parseRule (ruleNode, m_inverseRules);
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
    
    for (lat::StringList::const_iterator protocol = m_protocols.begin ();
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
replaceWithRegexp (const lat::RegexpMatch matches, 
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
	assert( i<1000000 );
	sprintf (buffer, "%i", i);
	std::string variableRegexp = std::string ("[$]") + buffer;
	std::string matchResult = matches.matchString (inputString, i);
	
	lat::Regexp sustitutionToken (variableRegexp);
	
	//std::cerr << "Current match: " << matchResult << std::endl;
	
	result = lat::StringOps::replace (result, 
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
	    
	lat::RegexpMatch matches;
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
    
    for (lat::StringList::const_iterator protocol = m_protocols.begin ();
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
    lat::Regexp grammar ("(lfname|guid)='(.*)'");
    lat::RegexpMatch grammarMatches;
    
    grammar.match (query, 0, 0, &grammarMatches);
    
    if (grammarMatches.numMatches () != 3)
    {
	throw FCTransactionException
	    ("TrivialFileCatalog::retrievePFN",
	     "malformed query. the only supported one is lfname='something'"
	     " or guid='something'");
    }
    
    std::string lfn = grammarMatches.matchString (query, 2);
    
    for (lat::StringList::iterator protocol = m_protocols.begin ();
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

    lat::Regexp grammar ("(pfname|guid)='(.*)'");
    lat::RegexpMatch grammarMatches;
    
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
    

    for (lat::StringList::iterator protocol = m_protocols.begin ();
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
