/*  common utilities of the CondCore Python buiding
 *
 */

#include<memory>
#include<string>

namespace cond {

  class DBSession;

  namespace impl {
    struct FWMagic;
  }

  // initialize framework
  class FWIncantation {
  public:
    FWIncantation();
    
  private:
    std::auto_ptr<impl::FWMagic> magic;
  };

  // a readonly CondDB and its transaction
 class CondDB {
  public:
    CondDB(DBSession * isession=0);
   ~CondDB();
   const char * allTags() const;
   
 private:
   DBSession * session;
 };

  // initializ cond, coral etc
  class RDBMS {
  public:
    RDBMS();
    ~RDBMS();
    explicit RDBMS(std::string const & authPath);
    explicit RDBMS(std::string const & user,std::string const & pass);

    CondDB getDB(std::string const & db);

  private:
    std::auto_ptr<DBSession> session;
  };


}
