#ifndef COND_DBCommon_RefBase_H
#define COND_DBCommon_RefBase_H
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
  class RefBase{
  public:
    // default constructor
    RefBase();
    explicit RefBase( cond::PoolTransaction& pooldb );
    // from externalized token
    RefBase( cond::PoolTransaction& pooldb, 
	     const std::string& token,
	     const std::string& className);
    // from type info
    RefBase( cond::PoolTransaction& pooldb, 
	     const std::string& token,
	     const std::type_info& refType );
    // copy constructor
    RefBase(const RefBase& aCopy);
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
    RefBase& operator=(const RefBase&);
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
    virtual ~RefBase();
  protected:
    pool::IDataSvc* m_datasvc;
    pool::Placement* m_place;
    // wrap pool smart pointer
    pool::RefBase m_content;
  };
}
#endif
