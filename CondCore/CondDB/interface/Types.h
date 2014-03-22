#ifndef CondCore_CondDB_Types_h
#define CondCore_CondDB_Types_h
//
// Package:     CondDB
// 
// 
/*
   Description: various foundation types for Conditions
*/
//
// Author:      Giacomo Govi
// Created:     June 2013
//

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/functional/hash.hpp>
//
#include "CondCore/CondDB/interface/Time.h"

namespace cond {

  // to be removed after the transition to new DB
  typedef enum { UNKNOWN_DB=0, COND_DB, ORA_DB } BackendType;
  static constexpr BackendType DEFAULT_DB = ORA_DB;

  typedef enum { 
    SYNCHRONIZATION_UNKNOWN = -1,
    OFFLINE=0, 
    HLT, 
    PROMPT, 
    PCL 
  } SynchronizationType;

  std::string synchronizationTypeNames( SynchronizationType type );

  SynchronizationType synchronizationTypeFromName( const std::string& name );

  typedef std::string Hash;
  static constexpr unsigned int HASH_SIZE = 40;

  // Basic element of the IOV sequence.
  struct Iov_t {
    virtual void clear();
    bool isValid() const;
    bool isValidFor( Time_t target ) const;
    Time_t since;
    Time_t till;
    Hash payloadId;
  };

  struct Tag_t {
    virtual void clear();
    std::string tag;
    std::string payloadType;
    TimeType timeType;
    Time_t endOfValidity;
    Time_t lastValidatedTime;
  };

  struct TagInfo_t {
    // FIX ME: to be simplyfied, currently keeping the same interface as CondCore/DBCommon/interface/TagInfo.h
    TagInfo_t(): name(""),token(""),lastInterval(0,0), lastPayloadToken(""),size(0){}
    std::string name;
    std::string token;
    cond::ValidityInterval lastInterval;
    std::string lastPayloadToken;
    size_t size;
  };

  // temporarely, to minimize changes in the clients code
  typedef TagInfo_t TagInfo;

  struct TagMetadata_t {
    SynchronizationType synchronizationType;
    std::string description;
    boost::posix_time::ptime insertionTime;
    boost::posix_time::ptime modificationTime;
  };

  // temporarely replacement for cond::LogDBEntry
  struct LogDBEntry_t {
    unsigned long long logId;
    std::string destinationDB;   
    std::string provenance;
    std::string usertext;
    std::string iovtag;
    std::string iovtimetype;
    unsigned int payloadIdx;
    unsigned long long lastSince;
    std::string payloadClass;
    std::string payloadToken;
    std::string exectime;
    std::string execmessage;
  };

  class GTEntry_t {
  public: 
    GTEntry_t():
      m_data(){
    }
    GTEntry_t( const std::tuple<std::string,std::string,std::string>& gtEntryData ):
      m_data( gtEntryData ){
    }
    GTEntry_t( const GTEntry_t& rhs ):
      m_data( rhs.m_data ){
    }

    GTEntry_t& operator=( const GTEntry_t& rhs ){
      m_data = rhs.m_data;
      return *this;
    }

    const std::string& recordName() const {
      return std::get<0>(m_data);
    }
    const std::string& recordLabel() const {
      return std::get<1>(m_data);
    }
    const std::string& tagName() const {
      return std::get<2>(m_data);
    }
    std::size_t hashvalue()const{
      // taken from TagMetadata existing implementation. 
      // Is it correct ordering by tag? Tags are not unique in a GT, while record+label are...
      boost::hash<std::string> hasher;
      std::size_t result=hasher(tagName());
      return result;
    }
    bool operator<(const GTEntry_t& toCompare ) const {
      return this->hashvalue()<toCompare.hashvalue();
    }

  private:
    std::tuple<std::string,std::string,std::string> m_data; 
  };


}

#endif
  
