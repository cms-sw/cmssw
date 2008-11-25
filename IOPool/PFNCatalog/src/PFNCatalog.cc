/* 
   Concrete implementation of a FileCatalog.
   Author: Dave Dykstra
 */

#include <string>
#include <stdexcept>
#ifndef POOL_PFNCATALOG_H
#include "PFNCatalog.h"
#endif
#include "FileCatalog/FCException.h"
#include "Reflex/PluginService.h"
using namespace pool;
PLUGINSVC_FACTORY_WITH_ID(PFNCatalog,std::string("pfncatalog"),FCImpl*())

pool::PFNCatalog::PFNCatalog ()
{  
}

pool::PFNCatalog::~PFNCatalog ()
{
}


void
pool::PFNCatalog::connect ()
{
    std::string mem ("memory://");
    std::string::size_type idx = m_url.find (mem);
    if (idx == std::string::npos)
    {
	throw FCTransactionException
	    ("PFNCatalog::connect",
	     ": Malformed url for PFN catalog, expect pfncatalog_memory://"); 
    }
    m_url = m_url.erase (0, idx + mem.length());	
    idx = m_url.find ("?");
    if (idx == std::string::npos)
    {
	throw FCTransactionException
	    ("PFNCatalog::connect",
	     ": Malformed url for PFN catalog, expect pfncatalog_memory://fileType?PFN"); 
    }
    m_filetype = m_url.substr(0, idx);
    m_url = m_url.erase (0, idx + 1);	
}

void
pool::PFNCatalog::disconnect () const
{
}

void
pool::PFNCatalog::start () const
{
}

void
pool::PFNCatalog::commit (FileCatalog::CommitMode /*cm=FileCatalog::REFRESH*/) const
{
}

void
pool::PFNCatalog::rollback () const
{
    throw FCTransactionException
	("PFNCatalog::rollback",
	 "PFN catalogs cannot rollback");
}
  
void
pool::PFNCatalog::registerPFN (const std::string& /*pfn*/, 
				       const std::string& /*filetype*/,
				       FileCatalog::FileID& /*fid*/) const
{
    throw FCTransactionException
	("PFNCatalog::registerPFN",
	 "It does not make sense to register PFN for PFNCatalogs");    
}   

void
pool::PFNCatalog::registerLFN (const std::string& /*pfn*/, 
				       const std::string& /*lfn*/) const
{
    throw FCTransactionException
	("PFNCatalog::registerLFN",
	 "It does not make sense to register LFN for PFNCatalogs");    
}

void
pool::PFNCatalog::addReplicaPFN (const std::string& /*pfn*/, 
					 const std::string& /*rpf*/) const
{
    throw FCTransactionException
	("PFNCatalog::addReplicalPFN",
	 "It does not make sense to register PFN for PFNCatalogs");    
}

void
pool::PFNCatalog::addPFNtoGuid (const FileCatalog::FileID& /*guid*/, 
					const std::string& /*pf*/, 
					const std::string& /*filetype*/) const
{
    throw FCTransactionException
	("PFNCatalog::addPFNtoGuid",
	 "It does not make sense to register replicas for PFNCatalogs");    
}

void
pool::PFNCatalog::renamePFN (const std::string& /*pfn*/, 
				     const std::string& /*newpfn*/) const
{
    throw FCTransactionException
	("PFNCatalog::renamePFN",
	 "It does not make sense to rename PFNs for PFNCatalogs");    
}

void
pool::PFNCatalog::lookupFileByPFN (const std::string & /*pfn*/, 
					   FileCatalog::FileID & fid, 
					   std::string& filetype) const
{
    fid = "00000000-0000-0000-0000-000000000000";
    filetype = m_filetype;
}

void
pool::PFNCatalog::lookupFileByLFN (const std::string& /*lfn*/, 
					   FileCatalog::FileID& /*fid*/) const
{
    throw FCTransactionException
	("PFNCatalog::lookupFileByLFN",
	 "looking up files by LFN is not supported for PFNCatalogs");    
}

void
pool::PFNCatalog::lookupBestPFN (const FileCatalog::FileID& /*fid*/, 
					 const FileCatalog::FileOpenMode& /*omode*/,
					 const FileCatalog::FileAccessPattern& /*amode*/,
					 std::string& pfn,
					 std::string& filetype) const
{
    pfn = m_url;
    filetype = m_filetype;
} 

void
pool::PFNCatalog::insertPFN (PFNEntry& /*pentry*/) const
{
    throw FCTransactionException
	("PFNCatalog::insertPFN",
	 "It does not make sense to insert PFNs for PFNCatalogs");    
}

void
pool::PFNCatalog::insertLFN (LFNEntry& /*lentry*/) const
{
    throw FCTransactionException
	("PFNCatalog::insertLFN",
	 "It does not make sense to insert LFNs for PFNCatalogs");    
}

void
pool::PFNCatalog::deletePFN (const std::string& /*pfn*/) const
{
    throw FCTransactionException
	("PFNCatalog::deletePFN",
	 "It does not make sense to delete PFNs for PFNCatalogs");    
}

void
pool::PFNCatalog::deleteLFN (const std::string& /*lfn*/) const
{
    throw FCTransactionException
	("PFNCatalog::deleteLFN",
	 "It does not make sense to delete LFNs for PFNCatalogs");    
}

void
pool::PFNCatalog::deleteEntry (const FileCatalog::FileID& /*guid*/) const
{
    throw FCTransactionException
	("PFNCatalog::deleteEntry",
	 "It does not make sense to delete GUIDs for PFNCatalogs");    
}

bool
pool::PFNCatalog::isReadOnly () const
{
    return true;    
}
 
bool
pool::PFNCatalog::retrievePFN (const std::string& /*query*/, 
				       FCBuf<PFNEntry>& /*buf*/, 
				       const size_t& /*start*/)
{
    throw FCTransactionException
	("PFNCatalog::retrievePFN",
	 "retrieving PFN is not supported for PFNCatalogs");    
    
    return false;    
}

bool
pool::PFNCatalog::retrieveLFN (const std::string& /*query*/, 
				       FCBuf<LFNEntry>& /*buf*/, 
				       const size_t& /*start*/)
{    
    throw FCTransactionException
	("PFNCatalog::retrieveLFN",
	 "retrieving LFN is not supported for PFNCatalogs");    
    
    return false;    
}

bool
pool::PFNCatalog::retrieveGuid (const std::string& /*query*/, 
					FCBuf<FileCatalog::FileID>& /*buf*/, 
					const size_t& /*start*/)
{   
    throw FCTransactionException
	("PFNCatalog::retrieveGuid",
	 "retrieving Guid is not supported for PFNCatalogs");    
    
    return false;    
}
