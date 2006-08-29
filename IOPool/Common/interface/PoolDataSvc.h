#ifndef Common_PoolDataSvc_h
#define Common_PoolDataSvc_h
//////////////////////////////////////////////////////////////////////
//
// $Id: PoolDataSvc.h,v 1.3 2006/08/07 22:07:25 wmtan Exp $
//
// Class PoolDataSvc. Common services to manage POOL cache
//
// Author: Bill Tanenbaum and Luca Lista
//
//////////////////////////////////////////////////////////////////////

#include <string>
#include "boost/shared_ptr.hpp"

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

    pool::IDataSvc *context() const {return context_.get();}
    
  private: 
    template <typename T>
    T
    getAttribute(std::string const& attributeName, std::string const& fileName) const;

    template <typename T>
    void
    setAttribute(std::string const& attributeName, std::string const& fileName, T const& value) const;

    boost::shared_ptr<pool::IDataSvc> context_;
  };
}

#endif
