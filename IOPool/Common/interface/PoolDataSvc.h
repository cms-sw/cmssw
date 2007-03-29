#ifndef IOPool_Common_PoolDataSvc_h
#define IOPool_Common_PoolDataSvc_h
//////////////////////////////////////////////////////////////////////
//
// $Id: PoolDataSvc.h,v 1.4 2006/08/29 22:49:36 wmtan Exp $
//
// Class PoolDataSvc. Common services to manage POOL cache
//
// Author: Bill Tanenbaum and Luca Lista
//
//////////////////////////////////////////////////////////////////////

#include "boost/shared_ptr.hpp"

namespace pool {
  class IDataSvc;
}

namespace edm {
  class InputFileCatalog;
  class OutputFileCatalog;
  class PoolDataSvc {
  public:
    PoolDataSvc(InputFileCatalog & catalog_, bool delete_on_free);
    PoolDataSvc(OutputFileCatalog & catalog_, bool delete_on_free);
    ~PoolDataSvc() {}

    pool::IDataSvc *context() const {return context_.get();}
    
  private: 
    boost::shared_ptr<pool::IDataSvc> context_;
  };
}
#endif
