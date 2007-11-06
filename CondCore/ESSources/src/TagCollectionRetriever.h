#ifndef CondCore_ESSources_TagCollectionRetriever_h
#define CondCore_ESSources_TagCollectionRetriever_h
//
// Package:    CondCore/ESSources
// Class:      TagCollectionRetriever
//
/**\class TagCollectionRetriever TagCollectionRetriever.h CondCore/ESSources/interface/TagCollectionRetriever.h
 Description: utility class to retrieve tag collection from db given the root tag
*/
//
// Author:      Zhen Xie
//
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
