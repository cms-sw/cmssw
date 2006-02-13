#ifndef COND_DBWRITER_H
#define COND_DBWRITER_H
#include <string>
#include "DataSvc/Ref.h"
#include "SealKernel/Exception.h"
#include "CondCore/DBCommon/interface/Exception.h"
namespace cond{
  class DBSession;
  class DBWriter{
  public:
    DBWriter( cond::DBSession& session,
	      const std::string& containerName );
    ~DBWriter();
    /**pin the object into the object cache
     */
    template<typename ObjType>
    std::string markWrite( ObjType* obj ){
      pool::Ref<ObjType> myref(&(m_session.DataSvc()), obj);
      try{
	myref.markWrite(*m_placement);
	++m_counter;
	
      }catch( const seal::Exception& er){
	throw cond::Exception( "caught seal::Exception "+ er.what() );
      }catch ( const std::exception& er ) {
	throw cond::Exception( "caught std::exception "+ er.what() );
      }catch ( ... ) {
	throw cond::Exception( "caught unknown exception ");
      }
      return myref.toString();
    }
  private:
    cond::DBSession& m_session;
    const std::string m_containerName;
    pool::Placement* m_placement;
    std::string m_mappingoutput;
  };
}
#endif
// DBWRITER_H













