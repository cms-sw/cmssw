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
//
#include "CondCore/CondDB/interface/Time.h"

namespace cond {

  typedef enum { 
    SYNCHRONIZATION_UNKNOWN = -1,
    OFFLINE=0, 
    HLT, 
    PROMPT, 
    PCL 
  } SynchronizationType;

  std::string synchronizationTypeNames( SynchronizationType type );

  SynchronizationType synchronizationTypeFromName( const std::string& name );

  template <typename T> 
  std::pair<const std::string,T> enumPair( std::string name, T value ){
    return std::make_pair( name, value );
  }

  typedef std::string Hash;
  static constexpr unsigned int HASH_SIZE = 40;

  // Basic element of the IOV sequence.
  struct Iov_t {
    void clear();
    bool isValid() const;
    bool isValidFor( Time_t target ) const;
    Time_t since;
    Time_t till;
    Hash payloadId;
  };

  struct Tag_t {
    void clear();
    std::string tag;
    std::string payloadType;
    TimeType timeType;
    Time_t endOfValidity;
    Time_t lastValidatedTime;
  };

  struct TagMetadata_t {
    SynchronizationType synchronizationType;
    std::string description;
    boost::posix_time::ptime insertionTime;
    boost::posix_time::ptime modificationTime;
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
  private:
    std::tuple<std::string,std::string,std::string> m_data; 
  };


}

#endif
  
