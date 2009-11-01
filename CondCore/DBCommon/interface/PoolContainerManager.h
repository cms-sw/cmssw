#ifndef CondCore_DBCommon_PoolContainerManager_H
#define CondCore_DBCommon_PoolContainerManager_H
#include "CondCore/DBCommon/interface/ContainerIterator.h"
#include <vector>
#include <string>
namespace cond{
  class PoolTransaction;
  class PoolContainerManager{
  public:
    explicit PoolContainerManager(  PoolTransaction& pooldb );
    void listAll( std::vector<std::string>& containers );
    template<typename T>
      ContainerIterator<T>* newContainerIterator(const std::string& containername);
    void exportContainer( PoolTransaction& destdb, 
			  const std::string& containername,
			  const std::string& className);
  private:
    cond::PoolTransaction* m_pooldb;
  };
}
template<typename T>
cond::ContainerIterator<T>* cond::PoolContainerManager::newContainerIterator(const std::string& containername){
  return new cond::ContainerIterator<T>(*m_pooldb, containername);
    }
#endif
