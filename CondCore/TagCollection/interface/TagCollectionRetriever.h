#ifndef CondCore_TagCollection_TagCollectionRetriever_h
#define CondCore_TagCollection_TagCollectionRetriever_h
//
// Package:    CondCore/TagCollection
// Class:      TagCollectionRetriever
//
/**\class TagCollectionRetriever TagCollectionRetriever.h CondCore/TagCollection/interface/TagCollectionRetriever.h
 Description: utility class to retrieve tag collection from db with given tag tree and node 
*/
//
// Author:      Zhen Xie
//

#include <set>
#include <string>
#include "CondCore/DBCommon/interface/TagMetadata.h"
#include "CondCore/TagCollection/interface/PfnEditor.h"
#include "CondCore/DBCommon/interface/DbSession.h"
namespace cond{
  class TagCollectionRetriever{
  public:
    /// constructor
    explicit TagCollectionRetriever( cond::DbSession& coraldb );
    explicit TagCollectionRetriever( cond::DbSession& coraldb, 
				     std::string const & prefix, 
				     std::string const & postfix);
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
    PfnEditor pfnEditor;
    
  };
}//ns cond
#endif
