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
#include <xercesc/dom/DOMCharacterData.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/framework/LocalFileFormatTarget.hpp>
#include <xercesc/util/XMLUni.hpp>
#include <xercesc/util/XMLURL.hpp>

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
      m_fileType ("ROOT_All")
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

	std::ifstream configFile;
	configFile.open (m_url.c_str ());

    
	trivialLog << seal::Msg::Info
		   << "Using catalog configuration " 
		   << m_url << seal::endmsg;

	if (!configFile.good () || !configFile.is_open ())
	{
	    m_transactionsta = 0;
            return;
	}

	configFile.close ();
	
	XercesDOMParser* parser = new XercesDOMParser;     
	parser->setValidationScheme(XercesDOMParser::Val_Auto);
	parser->setDoNamespaces(false);
	parser->parse(m_url.c_str());	
	DOMDocument* doc = parser->getDocument();
	ASSERT (doc);
	
	/* trivialFileCatalog matches the following xml schema
	   FIXME: write a proper DTD
	    <trivialCatalog>
	    <rule match="lfn/guid match regular expression">
	    <prefix>/foo/bar</prefix>
	    </rule>
	    </trivialCatalog>
	 */

	unsigned int ruleTagsNum  = 
	    doc->getElementsByTagName(_toDOMS("rule"))->getLength();
	
	// FIXME: we should probably use a DTD for checking validity 

	for (unsigned int i=0; i<ruleTagsNum; i++){
	    DOMNode* ruleNode = 
		doc->getElementsByTagName(_toDOMS("rule"))->item(i);
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
	    
	    std::string regExp = _toString (ruleElement->getAttribute (_toDOMS ("match")));
	    
	    DOMNodeList *prefixes 
		= ruleElement->getElementsByTagName (_toDOMS ("prefix"));
	    
	    
	    if (prefixes->getLength () != 1)
	    {
		throw FCTransactionException
		    ("TrivialFileCatalog::connect",
		     ":Malformed trivial catalog"); 		
	    }

	    DOMElement *prefixNode = dynamic_cast <DOMElement *> (prefixes->item (0));
	    if (!prefixNode)
	    {
		throw FCTransactionException
		    ("TrivialFileCatalog::connect",
		     ":Malformed trivial catalog"); 		
	    }

	    
	    DOMText *prefixText = dynamic_cast <DOMText *> (prefixNode->getFirstChild ());
	    if (!prefixText)
	    {
		throw FCTransactionException
		    ("TrivialFileCatalog::connect",
		     ":Malformed trivial catalog"); 		
	    }
	  
	    std::string prefix = _toString (prefixText->getData ());
	    
	    Rule rule;
	    rule.first = new seal::Regexp (regExp);
	    rule.first->compile ();
	    rule.second = prefix;
	    m_rules.push_back (rule);
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
    
    for (Rules::const_iterator i = m_rules.begin ();
	 i != m_rules.end ();
	 i++)
    {
	const std::string &prefix = i->second;
	std::string prefixUsed = pfn.substr (0, prefix.size ());
	ASSERT (prefixUsed.size () == prefix.size ());
	if (prefixUsed == prefix)
	{
	    fid = pfn.substr (prefix.size ());	    
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
    for (Rules::const_iterator i = m_rules.begin ();
	 i != m_rules.end ();
	 i++)
    {
	if (i->first->exactMatch (fid))
	{
	    pfn = i->second + fid; 
	    return;
	}
    }
    throw FCTransactionException
	("TrivialFileCatalog::lookupBestPFN",
	 "No match found in the TrivialFileCatalog");    
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
    seal::StringList tokens = seal::StringOps::split (query, "=");
    
    if (tokens.size () != 2)
    {
	throw FCTransactionException
	    ("TrivialFileCatalog::retrievePFN",
	     "malformed query. the only supported one is lfname='something'"
	     " or guid='something'");
    }
    
    if (tokens[0] != "lfname" 
	&& tokens[0] != "guid")
    {
	throw FCTransactionException
	    ("TrivialFileCatalog::retrievePFN",
	     "malformed query. the only supported one is lfname='something'"
	     " or guid='something'");    
    }
    
    std::string lfn = seal::StringOps::remove (tokens[1], "'");
    
    std::string fidCandidate;
    
    for (Rules::iterator i = m_rules.begin ();
	 i != m_rules.end ();
	 i++)
    {
	if (i->first->exactMatch (lfn))
	{
	    buf.push_back (PFNEntry(i->second + lfn, 
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
pool::TrivialFileCatalog::retrieveLFN (const std::string& /*query*/, 
				       FCBuf<LFNEntry>& /*buf*/, 
				       const size_t& /*start*/)
{    
    throw FCTransactionException
	("TrivialFileCatalog::retrieveGuid",
	 "Cannot retrieve GUIDs with TrivialCatalogs");
}

bool
pool::TrivialFileCatalog::retrieveGuid (const std::string& /*query*/, 
					FCBuf<FileCatalog::FileID>& /*buf*/, 
					const size_t& /*start*/)
{   
    throw FCTransactionException
	("TrivialFileCatalog::retrieveGuid",
	 "Cannot retrieve GUIDs with TrivialCatalogs");
}
