#ifndef CondCore_Utilities_CondPyInterface_h
#define CondCore_Utilities_CondPyInterface_h

/*  common utilities of the CondCore Python buiding
 *
 */

#include "CondCore/IOVService/interface/IOVProxy.h"


#include<boost/shared_ptr.hpp>
#include<string>

namespace cond {

  class DBSession;
  class Connection;

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
   CondDB(Connection * conn);
   ~CondDB();
   const char * allTags() const;

   IOVProxy iov(std::string & tag) const;

   
 private:
   mutable Connection * me;
 };

  // initializ cond, coral etc
  class RDBMS {
  public:
    RDBMS();
    ~RDBMS();
    explicit RDBMS(std::string const & authPath);
    RDBMS(std::string const & user,std::string const & pass);

    CondDB getDB(std::string const & db);

  private:
    boost::shared_ptr<DBSession> session;
  };


}


#endif //  CondCore_Utilities_CondPyInterface_h
