/*  common utilities of the CondCore Python buiding
 *
 */

#include<boost/shared_ptr>
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
   CondDB(Connection * conn);
   ~CondDB();
   const char * allTags() const;
   
 private:
   Connection * me;
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
