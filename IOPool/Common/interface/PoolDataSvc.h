#ifndef Common_PoolDataSvc_h
#define Common_PoolDataSvc_h
//////////////////////////////////////////////////////////////////////
//
// $Id: PoolDataSvc.h,v 1.2 2006/04/06 23:45:51 wmtan Exp $
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

    size_t getFileSize(std::string const& fileName) const;

    void setCompressionLevel(std::string const& fileName, int value) const;

    pool::IDataSvc *context() {return context_;}
    
  private: 
    template <typename T>
    T
    getAttribute(std::string const& attributeName, std::string const& fileName) const;

    template <typename T>
    void
    setAttribute(std::string const& attributeName, std::string const& fileName, T const& value) const;

    pool::IDataSvc *context_;
  };
}

#endif
