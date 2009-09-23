#ifndef CondCore_ESSources_TagCollectionRetriever_h
#define CondCore_ESSources_TagCollectionRetriever_h
//
// Package:    CondCore/ESSources
// Class:      TagCollectionRetriever
//
/**\class TagCollectionRetriever TagCollectionRetriever.h CondCore/ESSources/interface/TagCollectionRetriever.h
 Description: utility class to retrieve tag collection from db with given tag tree and node 
*/
//
// Author:      Zhen Xie
// $Id: TagCollectionRetriever.h,v 1.5 2008/08/22 17:07:19 xiezhen Exp $
//

#include <set>
#include <string>
#include "CondCore/DBCommon/interface/TagMetadata.h"
#include "CondCore/DBCommon/interface/DbSession.h"
namespace cond{
  class CoralTransaction;
  class TagCollectionRetriever{
  public:
    /// constructor
    explicit TagCollectionRetriever( cond::DbSession& coraldb );
    /// destructor
    ~TagCollectionRetriever();
    /**
       given global tag return the basic tag collection. The global tag has the format TreeName::NodeName
    */
    void getTagCollection( const std::string& globaltag,
                           std::set<cond::TagMetadata>& result);
  private:
    /// parse global tag string returns result in pair <treename,nodename>
    std::pair<std::string,std::string> parseglobaltag(const std::string& globaltag);
    cond::DbSession m_coraldb;
  };
}//ns cond
#endif
