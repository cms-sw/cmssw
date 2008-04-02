#ifndef COND_DBCommon_GenericRef_H
#define COND_DBCommon_GenericRef_H
#include <string>
#include <typeinfo>
#include "DataSvc/RefBase.h"
namespace pool{
  class IDataSvc;
}
namespace cond{
  class PoolTransaction;
  /* 
     adapter to pool::RefBase defines ref interface
  **/
  class GenericRef{
  public:
    // default constructor
    GenericRef();
    explicit GenericRef( cond::PoolTransaction& pooldb );
    // from externalized token
    GenericRef( cond::PoolTransaction& pooldb, 
		const std::string& token);
    // from externalized token and class name
    GenericRef( cond::PoolTransaction& pooldb, 
		const std::string& token,
		const std::string& className);
    // from type info
    GenericRef( cond::PoolTransaction& pooldb, 
		const std::string& token,
		const std::type_info& refType );
    // copy constructor
    GenericRef(const GenericRef& aCopy);
    /* Query interface
    **/
    // externalised token
    const std::string token() const;
    // object name
    std::string className() const;
    // container name
    std::string containerName() const;
    /* update operations
    **/
    // register for write
    void markWrite(const std::string& container);
    // register for update 
    void markUpdate();
    // register for delete
    void markDelete();
    // RefBase assignment operator 
    GenericRef& operator=(const GenericRef&);
    // default copy operator (defined copy policy)
    //void copy(const RefBase&);
    // copy the entire RefBase content, with type checking 
    //void copyDeep(const RefBase&);
    // copy only the object pointer, with type checking 
    //void copyShallow(const RefBase&);
    // export to another destination db; 
    std::string exportTo( cond::PoolTransaction& destdb );
    // reset the whole ref content
    void clear();
    // explicitly delete the object pointee
    void reset();
    // destructor
    virtual ~GenericRef();
  protected:
    pool::IDataSvc* m_datasvc;
    pool::Placement* m_place;
    // wrap pool smart pointer
    pool::RefBase m_data;
  };
}
#endif
