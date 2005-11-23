#ifndef Common_PoolDataSvc_h
#define Common_PoolDataSvc_h
//////////////////////////////////////////////////////////////////////
//
// $Id: PoolDataSvc.h,v 1.1 2005/11/01 22:42:45 wmtan Exp $
//
// Class PoolDataSvc. Common services to manage POOL cache
//
// Author: Bill Tanenbaum and Luca Lista
//
//////////////////////////////////////////////////////////////////////

#include <string>

namespace pool {
  class IDataSvc;
  class IDataBase;
}

namespace edm {
  class PoolCatalog;
  class PoolDataSvc {
  public:
    PoolDataSvc(PoolCatalog & catalog_, bool write, bool delete_on_free);
    ~PoolDataSvc() {}
    size_t getFileSize(std::string const& fileName);
    pool::IDataSvc *context() {return context_;}
    
  private: 
    pool::IDataSvc *context_;
  };
}

#endif
