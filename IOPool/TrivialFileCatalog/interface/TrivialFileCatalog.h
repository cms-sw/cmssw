#ifndef POOL_TRIVIALFILECATALOG_H
#define POOL_TRIVIALFILECATALOG_H
#ifndef POOL_FCIMPL_H
#include "FileCatalog/FCImpl.h"
#endif
#ifndef FCBUF_H
#include "FileCatalog/FCBuf.h"
#endif
#include <list>
#include <utility>
#include <SealBase/Regexp.h>
#include <xercesc/dom/DOM.hpp>

namespace pool 
{
/**
 *        @class TrivialFileCatalog TrivialFileCatalog.h TrivialCatalog/TrivialFileCatalog.h 
 *          	This class is the concrete implementation of the 
 *          	POOL Trivial File Catalog as requested by Peter Elmer.
 *	    	A TrivialCatalog is one where the LFN and the GUID are the same and the PFN
 *	    	is generated from the LFN-GUID according to some configuration.
 *	        The following constraints are imposed by such a kind of catalog.
 *		 - Quering the catalog for anything but a given LFN/GUID does not make sense.
 *		 - Registering files to the catalog makes no sense.
 *		 - No metadata is allowed.
 *		 - No replicas or aliases.
 *       @Author: Giulio Eulisse
 */

    

class TrivialFileCatalog : public FCImpl 
{
public:
    TrivialFileCatalog ();
    virtual ~TrivialFileCatalog ();
    //
    //Connection and transaction control operations.
    //   

    /** TrivialFileCatalog connection method.
	@param url [IN] std::string standard URL of the TrivialCatalog
	configuration file.
    */
    virtual void connect();

    /** Close the connection to the TrivialFileCatalog.
     */
    virtual void disconnect() const;
    /** Start the catalog transaction.
     */
    virtual void start() const;
    /** Commit the catalog transaction.
	NOTE: since it makes no sense to do write operation on a trivial.

	@param cm [IN] catalog commit mode.
	REFRESH mode: catalog will be reinitialised on next transaction start.
	ONHOLD mode: catalog will not be reinitialised on next transaction start.
    */
    virtual void commit(FileCatalog::CommitMode cm=FileCatalog::REFRESH) const;

    /**Rollback the catalog transaction.
       NOTE: Does not make sense for the trivial catalog.
     */
    virtual void rollback() const;  

    //
    //File registration operations
    //
  
    /**Register a PFN when a new file is created, 
       NOTE: Does not make sense for the trivial catalog.

       returns the corresponding std::string 
       @param  pfn [IN] the Physical file name
       @param  filetype [IN] the filetype pf the PFN
       @param  fid [OUT] Guid of the file
    */
    virtual void registerPFN(const std::string& pfn, 
			     const std::string& filetype,
			     FileCatalog::FileID& fid ) const;   

    /** Register LFN 
	NOTE: Does not make sense for the trivial catalog.

	@param pfn PFN
	@param lfn LFN
    */
    virtual void registerLFN(const std::string& pfn, 
			     const std::string& lfn) const;


    /** Add a replica file name in the catalog.
	NOTE: Does not make sense for the trivial catalog.
	@param pfn [IN] the PFN of the master copy file.
	@param rpf [IN] the PFN of the replica to be added in the catalog. 
    */
    virtual void addReplicaPFN(const std::string& pfn, 
			       const std::string& rpf) const;

    /** Add a mapping in the catalog.
	NOTE: Does not make sense for the trivial catalog.
	@param guid [IN] the Guid of the file.
	@param pf   [IN] the PFN of the file to be added in the catalog. 
    */
    virtual void addPFNtoGuid(const FileCatalog::FileID& guid, 
			      const std::string& pf, 
			      const std::string& filetype) const;

    /** Rename a PFN in the catalog. The use case is the file is moved.
	NOTE: Does not make sense for the trivial catalog.
	@param pfn [IN] old PFN.
	@param newpf [IN] new PFN.
    */
    virtual void renamePFN(const std::string& pfn, 
			   const std::string& newpfn) const;
  
    /** Lookup the fileID with given PFN.
	NOTE: Could be implemented if the mapping function FileID/LFN->PFN 
	is invertible.
	@param pfn [IN] PFN.
	@param fid [OUT] FileID, return empty string if not found.
	@param ftype [OUT] file type , return empty string if not found.
    */
    virtual void lookupFileByPFN(const std::string& pfn, 
				 FileCatalog::FileID& fid, 
				 std::string& ftype) const;
  
    /** Lookup the FileID with given LFN.
	@param lfn [IN] LFN.
	@param fid [OUT] FileID.
    */
    virtual void lookupFileByLFN(const std::string& lfn, 
				 FileCatalog::FileID& fid) const;

    /** Lookup the PFN of a file with given std::string in the catalog.
	Throws exception when the file is nonexistant.
	@param fid [IN] FileID
	@param omode [IN] FileOpenMode(read,write,update).
	@param amode [IN] FileAccessPattern(sequential,random,partial random).
	A hint to decide on how to ship and/or replicate the file. 
	@param pfn [OUT] PFN.
	@param filetype [OUT] file type.
    */
    virtual void lookupBestPFN(const FileCatalog::FileID& fid, 
			       const FileCatalog::FileOpenMode& omode,
			       const FileCatalog::FileAccessPattern& amode,
			       std::string& pfn,
			       std::string& filetype) const; 
    /** Does not make sense for a Trivial catalog*/
    virtual void insertPFN(PFNEntry& pentry) const;
    /** Does not make sense for a Trivial catalog*/
    virtual void insertLFN(LFNEntry& lentry) const;
    /** Does not make sense for a Trivial catalog*/
    virtual void deletePFN(const std::string& pfn) const;
    /** Does not make sense for a Trivial catalog*/
    virtual void deleteLFN(const std::string& lfn) const;
    /** Does not make sense for a Trivial catalog*/
    virtual void deleteEntry(const FileCatalog::FileID& guid) const;

    virtual bool isReadOnly() const;
 
    /** Only a subset of the queries are supported*/
    virtual bool retrievePFN(const std::string& query, 
			     FCBuf<PFNEntry>& buf, 
			     const size_t& start );
    /** Only a subset of the queries are supported*/
    virtual bool retrieveLFN(const std::string& query, 
			     FCBuf<LFNEntry>& buf, 
			     const size_t& start );
    /** Only a subset of the queries are supported*/
    virtual bool retrieveGuid(const std::string& query, 
			      FCBuf<FileCatalog::FileID>& buf, 
			      const size_t& start );

private:
    /** For the time being the only allowed configuration item is a
     *  prefix to be added to the GUID/LFN.
     */ 
    mutable bool 	m_connectionStatus;
    static int		s_numberOfInstances;    

    
    
    typedef struct {
	seal::Regexp pathMatch;
	seal::Regexp destinationMatch;	
	std::string result;
	std::string chain;
    } Rule;

    typedef std::list <Rule> Rules;
    typedef std::map <std::string, Rules> ProtocolRules;

    /** API to add rules to the TrivialFileCatalog
        without having to specify them in the xml file.
        @param protocol [IN] protocol to which the rule is associated.
        @param destinationMatchRegexp [IN] regular expression matching the destination 
        @param pathMatchRegexp [IN] regular expression matching 
        @param result [IN] format of the transformed path
        @param chain [IN] what to chain this rule with
        @param direct [IN] wether the rule is to be used for direct (lnf-to-pfn)
                           or inverse (pfn-to-lfn) mapping.
        @param back [IN] wether the rule needs to be added at the end or at the front of
                         list of rules.
     */
    void addRule (const std::string &protocol,
                  const std::string &destinationMatchRegexp,
                  const std::string &pathMatchRegexp,
                  const std::string &result,
                  const std::string &chain,
                  bool direct=true, bool back=true);


    
    void parseRule (xercesc::DOMNode *ruleNode, bool direct);
    
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
    seal::StringList		m_protocols;
    std::string			m_destination;    
};    
    

}


#endif
