#ifndef CondCore_MetaDataService_METADATA_H
#define CondCore_MetaDataService_METADATA_H
#include <string>
#include <vector>
#include "CondCore/DBCommon/interface/Time.h"
#include "CondCore/DBCommon/interface/DbSession.h"
#include "MetaDataEntry.h"
//
// Package:     MetaDataService
// Class  :     MetaData
// 
/**\class MetaData MetaData.h CondCore/MetaDataService/interface/MetaDataService.h
   Description: class for handling IOV metadata
*/
//
// Author:      Zhen Xie
//
namespace cond{
  class MetaData {
  public:
    // constructor
    explicit MetaData( cond::DbSession& userSession );
    // destructor
    ~MetaData();
    // add metadata entry
    bool addMapping(const std::string& name, const std::string& token,cond::TimeType timetype=cond::runnumber);
    // replace iov token associated with a given tag
    //bool replaceToken(const std::string& name, const std::string& newtoken);
    // if given iov tag exists
    bool hasTag( const std::string& name ) const;
    // list all tags
    void listAllTags( std::vector<std::string>& result ) const;
    // list all entries in the metadata table
    //void listAllEntries( std::vector<cond::MetaDataEntry>& result ) const;
    // get iov token associated with given tag
    const std::string getToken( const std::string& tagname ) const;
    // get the metadata table entry by tag name
    //void getEntryByTag( const std::string& tagname, cond::MetaDataEntry& result )const;
    // delete all entries in the metadata table
    void deleteAllEntries();
    // delete metadata entry selected by iov token
    //void deleteEntryByToken( const std::string& token );
    // delete metadata entry selected by tag name
    void deleteEntryByTag( const std::string& tag );
  private:
    // create metadata table
    //void createTable(const std::string& tabname);
    mutable cond::DbSession m_userSession;
  };
}
#endif
 
