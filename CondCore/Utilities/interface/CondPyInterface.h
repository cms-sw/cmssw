#ifndef CondCore_Utilities_CondPyInterface_h
#define CondCore_Utilities_CondPyInterface_h

/*  common utilities of the CondCore Python buiding
 *
 */

#include "CondCore/IOVService/interface/IOVProxy.h"
#include "CondCore/DBCommon/interface/LogDBEntry.h"
#include "CondCore/DBCommon/interface/DbConnection.h"
#include "CondCore/DBCommon/interface/TagMetadata.h"


#include<boost/shared_ptr.hpp>
#include<string>
#include<set>

namespace cond {

  typedef std::set<cond::TagMetadata> GlobalTag;


  class Logger;

  namespace impl {
    struct FWMagic;
  }

  // initialize framework
  class FWIncantation {
  public:
    FWIncantation();
    // FWIncantation(FWIncantation const & other );
    ~FWIncantation();
    
  private:
    boost::shared_ptr<impl::FWMagic> magic;
  };

  // a readonly CondDB and its transaction
 class CondDB {
  public:
   CondDB();
   CondDB(const CondDB & other);
   CondDB & operator=(const CondDB & other);
   CondDB(DbSession& session, boost::shared_ptr<cond::Logger> ilog );
   ~CondDB();
   std::string allTags() const;

   IOVProxy iov(std::string const & tag) const;
   IOVProxy iovWithLib(std::string const & tag) const;

   IOVElementProxy payLoad(std::string const & token) const;

   std::string iovToken(std::string const & tag) const;
   
   cond::LogDBEntry lastLogEntry(std::string const & tag) const;
   cond::LogDBEntry lastLogEntryOK(std::string const & tag) const;

   void startTransaction() const;
   void startReadOnlyTransaction() const;
   void commitTransaction() const;
   
   void closeSession() const;
   
   DbSession& session() const { return me;}

 private:
   mutable DbSession me;
   boost::shared_ptr<cond::Logger> logger;
 };

  // initializ cond, coral etc
  class RDBMS {
  public:
    RDBMS();
    ~RDBMS();
    explicit RDBMS(std::string const & authPath, bool debug=false);
    RDBMS(std::string const & user,std::string const & pass);
    void setLogger(std::string const & connstr);

    CondDB getDB(std::string const & db);
    
    CondDB getReadOnlyDB(std::string const & db);
    
    GlobalTag const & globalTag(std::string const & connstr, 
				std::string const & gname,
				std::string const & prefix, 
				std::string const & postfix) const;

  private:
    boost::shared_ptr<DbConnection> connection;
    boost::shared_ptr<cond::Logger> logger;
    GlobalTag m_globalTag;

  };


}


#endif //  CondCore_Utilities_CondPyInterface_h
