#ifndef Common_PoolDataSvc_h
#define Common_PoolDataSvc_h
//////////////////////////////////////////////////////////////////////
//
// $Id: PoolDataSvc.h,v 1.1 2005/11/23 02:17:56 wmtan Exp $
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
  class InputFileCatalog;
  class OutputFileCatalog;
  class PoolDataSvc {
  public:
    PoolDataSvc(InputFileCatalog & catalog_, bool delete_on_free);
    PoolDataSvc(OutputFileCatalog & catalog_, bool delete_on_free);
    ~PoolDataSvc() {}
    size_t getFileSize(std::string const& fileName);
    pool::IDataSvc *context() {return context_;}
    
  private: 
    pool::IDataSvc *context_;
  };
}

#endif
