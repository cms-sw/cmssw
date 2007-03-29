#ifndef IOPool_Common_PoolDatabase_h
#define IOPool_Common_PoolDatabase_h
//////////////////////////////////////////////////////////////////////
//
// $Id: PoolDatabase.h,v 1.4 2006/08/29 22:49:36 wmtan Exp $
//
// Class PoolDatabase. Common services to manage POOL database (a.k.a. file)
//
// Author: Bill Tanenbaum
//
//////////////////////////////////////////////////////////////////////

#include <string>
#include "boost/shared_ptr.hpp"

namespace pool {
  class IDatabase;
}

namespace edm {
  class PoolDataSvc;
  class PoolDatabase {
  public:
    PoolDatabase() : fileName_(), databaseHandle_() {}
    PoolDatabase(std::string const& fileName, PoolDataSvc const& dataSvc);
    ~PoolDatabase() {}

    std::string const& fileName() const {return fileName_;}
    size_t getFileSize() const;
    void setCompressionLevel(int value) const;
    void setBasketSize(int value) const;
    void setSplitLevel(int value) const;


  private: 
    template <typename T>
    T
    getAttribute(std::string const& attributeName) const;

    template <typename T>
    void
    setAttribute(std::string const& attributeName, T const& value) const;

    std::string fileName_;
    boost::shared_ptr<pool::IDatabase> databaseHandle_;
  };
}

#endif
