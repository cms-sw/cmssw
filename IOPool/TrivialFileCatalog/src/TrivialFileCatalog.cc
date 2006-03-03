/* 
   Concrete implementation of a FileCatalog.
   Author: Giulio.Eulisse@cern.ch
 */

#include <set>
#include <exception>
#include <string>
#ifndef POOL_TRIVIALFILECATALOG_H
#include "IOPool/TrivialFileCatalog/interface/TrivialFileCatalog.h"
#endif
#include "POOLCore/PoolMessageStream.h"
#include "FileCatalog/FCException.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <SealBase/StringList.h>
#include <SealBase/StringOps.h>
#include <SealBase/DebugAids.h>

pool::TrivialFileCatalog::TrivialFileCatalog ()
    : m_connectionStatus (false),
      m_prefix ("/localscratch/data/"),
      m_fileType ("ROOT_All")
{
}

pool::TrivialFileCatalog::~TrivialFileCatalog ()
{
}


void
pool::TrivialFileCatalog::connect ()
{
    try
    {
	PoolMessageStream trivialLog( "TrivialFileCatalog", seal::Msg::Nil );
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

	char buffer[4096];
	if (!configFile.good () || !configFile.is_open ())
	{
	    m_transactionsta = 0;
            return;
	}
    
	while (! configFile.eof ())
	{
	    configFile.getline (buffer, 4096);
	    std::string line = buffer;
	
	    if (line != "")
	    {
		seal::StringList tokens = seal::StringOps::split (line, "=");

		if (tokens.size () != 2)
		{
		    throw FCTransactionException
			("TrivialFileCatalog::connect",
			 "Wrong configuration file"); 
		}	
	   
		std::string parameter = tokens[0];
		std::string value = tokens[1];
	
		if (parameter == "prefix")
		{
		    m_prefix = value;
		}
		else if (parameter == "type")
		{
		    m_fileType = value;
		}
	    }
	}    
	m_transactionsta = 1;
    }
    catch (seal::Exception& e)
    {
	m_transactionsta = 0;	
	throw FCconnectionException("TrivialFileCatalog::connect",e.message());
    }
    catch( std::exception& er)
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
				       FileCatalog::FileID& /*fid*/ ) const
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
    fid = pfn.substr(m_prefix.size());
}

void
pool::TrivialFileCatalog::lookupFileByLFN (const std::string& lfn, 
					   FileCatalog::FileID& fid) const
{
    if (m_transactionsta == 0)
	throw FCconnectionException("TrivialFileCatalog::lookupFileByLFN",
				    "Catalog not connected");
	

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
    pfn = m_prefix + fid;    
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
				       const size_t& /*start*/ )
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
    
    
    buf.push_back (PFNEntry(m_prefix + lfn, 
			    lfn, 
			    m_fileType));    
    return true;    
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
