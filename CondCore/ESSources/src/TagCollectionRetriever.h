#ifndef CondCore_ESSources_TagCollectionRetriever_h
#define CondCore_ESSources_TagCollectionRetriever_h
#include <map>
#include <string>
#include "CondCore/DBCommon/interface/TagMetadata.h"
namespace cond{
  class CoralTransaction;
  class TagCollectionRetriever{
  public:
    explicit TagCollectionRetriever( cond::CoralTransaction& coraldb );
    ~TagCollectionRetriever();
    void getTagCollection( const std::string& roottag,
			   std::map< std::string, cond::TagMetadata >& result);
  private:
    cond::CoralTransaction* m_coraldb;
  };
}
#endif
