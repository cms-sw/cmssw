#ifndef DBWRITER_H
#define DBWRITER_H
#include <string>
#include "DataSvc/Ref.h"
#include "SealKernel/Exception.h"
namespace pool{
  class IFileCatalog;
  class IDataSvc;
  class Placement;
}

namespace cond{
  class DBWriter{
  public:
    DBWriter(  const std::string& con );
    virtual ~DBWriter();
    /** start transaction
     */
    void startTransaction();
    /** commit transaction
     */
    void commitTransaction();
    bool containerExists(const std::string& containerName);
    void openContainer( const std::string& containerName );
    void createContainer( const std::string& containerName );    
    /**pin the object into the object cache
     */
    template<typename ObjType>
    std::string write( ObjType* obj ){
      pool::Ref<ObjType> myref(m_svc, obj);
      try{
	myref.markWrite(*m_placement);
      }catch( const seal::Exception& er){
	std::cout << er.what() << std::endl;    
      }catch ( const std::exception& er ) {
	std::cout << er.what() << std::endl;
      }catch ( ... ) {
	std::cout << "Funny error" << std::endl;
      }
      return myref.toString();
    }
  private:
    const std::string m_con;
    pool::IFileCatalog* m_cat;
    pool::IDataSvc* m_svc;
    pool::Placement* m_placement;
  };
}
#endif
// DBWRITER_H













