#ifndef CondCore_DBCommon_PoolContainerManager_H
#define CondCore_DBCommon_PoolContainerManager_H
#include "CondCore/DBCommon/interface/ContainerIterator.h"
#include "CondCore/DBCommon/interface/DbSession.h"
#include <vector>
#include <string>
namespace cond{
  class PoolContainerManager{
  public:
    explicit PoolContainerManager(  DbSession& pooldb );
    void listAll( std::vector<std::string>& containers );
    template<typename T>
      ContainerIterator<T>* newContainerIterator(const std::string& containername);
    void exportContainer( DbSession& destdb,
                          const std::string& containername,
                          const std::string& className);
  private:
    cond::DbSession m_pooldb;
  };
}
template<typename T>
cond::ContainerIterator<T>* cond::PoolContainerManager::newContainerIterator(const std::string& containername){
  return new cond::ContainerIterator<T>(m_pooldb, containername);
    }
#endif
