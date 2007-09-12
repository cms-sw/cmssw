#ifndef POOL_PFNCATALOG_H
#define POOL_PFNCATALOG_H
#ifndef POOL_FCIMPL_H
#include "FileCatalog/FCImpl.h"
#endif
#ifndef FCBUF_H
#include "FileCatalog/FCBuf.h"
#endif

namespace pool 
{
/**
 *        @class PFNCatalog PFNCatalog.h interface/PFNCatalog.h 
 *          	This class is a concrete implementation of the POOL
 *          	File Catalog.  A PFNCatalog is a catalog for applications
 *		that have no need for POOL catalog flexibility and just
 *		need a way to fit into POOL's catalog architecture.
 *		Each PFNCatalog object corresponds to one Physical File
 *		Name (a connect string), which is incorporated in the
 *		catalog connect string.  The format of the catalog
 *		connect string is
 *		    pfncatalog_memory://fileType?PFN
 *		where fileType is one of the types defined by pool::DbType.
 *		A lot of functions are required to be present in this
 *		by the parent class pool::FCImpl, but the only functions
 *		that actually do anything are connect(), lookupBestPFN(),
 *		and lookupFileByPFN().
 *       @Author: Dave Dykstra
 */

    

class PFNCatalog : public FCImpl 
{
public:
    PFNCatalog ();
    virtual ~PFNCatalog ();

    /** PFNCatalog connection method.
	@param url [IN] std::string standard URL of the PFNCatalog
	configuration file.
    */
    virtual void connect();

    virtual void disconnect() const;
    virtual void start() const;
    virtual void commit(FileCatalog::CommitMode cm=FileCatalog::REFRESH) const;
    virtual void rollback() const;  
    virtual void registerPFN(const std::string& pfn, 
			     const std::string& filetype,
			     FileCatalog::FileID& fid ) const;   
    virtual void registerLFN(const std::string& pfn, 
			     const std::string& lfn) const;
    virtual void addReplicaPFN(const std::string& pfn, 
			       const std::string& rpf) const;
    virtual void addPFNtoGuid(const FileCatalog::FileID& guid, 
			      const std::string& pf, 
			      const std::string& filetype) const;
    virtual void renamePFN(const std::string& pfn, 
			   const std::string& newpfn) const;
    /** Lookup the fileID with given PFN.
        NOTE: this is used for output.  The pfn parameter is ignored
	    and the same fid and ftype is returned for everything.
        @param pfn [IN] PFN.
        @param fid [OUT] FileID, return empty string if not found.
        @param ftype [OUT] file type , return empty string if not found.
    */
    virtual void lookupFileByPFN(const std::string& pfn, 
				 FileCatalog::FileID& fid, 
				 std::string& ftype) const;
    virtual void lookupFileByLFN(const std::string& lfn, 
				 FileCatalog::FileID& fid) const;

    /** Lookup the PFN of a file with given std::string in the catalog.
	Only the last two parameters are used for anything in PFNCatalog
	@param fid [IN] FileID
	@param omode [IN] FileOpenMode(read,write,update).
	@param amode [IN] FileAccessPattern(sequential,random,partial random).
	@param pfn [OUT] PFN.
	@param filetype [OUT] file type.
    */
    virtual void lookupBestPFN(const FileCatalog::FileID& fid, 
			       const FileCatalog::FileOpenMode& omode,
			       const FileCatalog::FileAccessPattern& amode,
			       std::string& pfn,
			       std::string& filetype) const; 
    virtual void insertPFN(PFNEntry& pentry) const;
    virtual void insertLFN(LFNEntry& lentry) const;
    virtual void deletePFN(const std::string& pfn) const;
    virtual void deleteLFN(const std::string& lfn) const;
    virtual void deleteEntry(const FileCatalog::FileID& guid) const;

    virtual bool isReadOnly() const;
 
    virtual bool retrievePFN(const std::string& query, 
			     FCBuf<PFNEntry>& buf, 
			     const size_t& start );
    virtual bool retrieveLFN(const std::string& query, 
			     FCBuf<LFNEntry>& buf, 
			     const size_t& start );
    virtual bool retrieveGuid(const std::string& query, 
			      FCBuf<FileCatalog::FileID>& buf, 
			      const size_t& start );

private:
    std::string m_filetype;

};
    

}


#endif
