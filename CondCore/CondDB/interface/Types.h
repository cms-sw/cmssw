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
#include <functional>
//
#include "CondCore/CondDB/interface/Time.h"

namespace cond {

  struct UserLogInfo {
    std::string provenance;
    std::string usertext;
  };

  typedef enum {
    SYNCH_ANY = 0,
    SYNCH_VALIDATION,
    SYNCH_OFFLINE,
    SYNCH_MC,
    SYNCH_RUNMC,
    SYNCH_HLT,
    SYNCH_EXPRESS,
    SYNCH_PROMPT,
    SYNCH_PCL
  } SynchronizationType;

  std::string synchronizationTypeNames(SynchronizationType type);

  SynchronizationType synchronizationTypeFromName(const std::string& name);

  typedef std::string Hash;
  static constexpr unsigned int HASH_SIZE = 40;

  // Basic element of the IOV sequence.
  struct Iov_t {
    Iov_t() : since(time::MAX_VAL), till(time::MIN_VAL), payloadId("") {}
    virtual ~Iov_t() = default;
    virtual void clear();
    bool isValid() const;
    bool isValidFor(Time_t target) const;
    Time_t since;
    Time_t till;
    Hash payloadId;
  };

  struct Tag_t {
    virtual ~Tag_t() = default;
    virtual void clear();
    std::string name;
    std::string payloadType;
    TimeType timeType;
    SynchronizationType synchronizationType;
    Time_t endOfValidity;
    Time_t lastValidatedTime;
  };

  struct TagInfo_t {
    // FIX ME: to be simplyfied, currently keeping the same interface as CondCore/DBCommon/interface/TagInfo.h
    TagInfo_t() : name(""), lastInterval() {}
    std::string name;
    Iov_t lastInterval;
    size_t size = 0;
    bool isEmpty() const { return lastInterval.since == time::MAX_VAL; }
  };

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

  struct GTMetadata_t {
    Time_t validity;
    std::string description;
    std::string release;
    boost::posix_time::ptime insertionTime;
    boost::posix_time::ptime snapshotTime;
  };

  class GTEntry_t {
  public:
    GTEntry_t() : m_data() {}
    GTEntry_t(const std::tuple<std::string, std::string, std::string>& gtEntryData) : m_data(gtEntryData) {}
    GTEntry_t(const GTEntry_t& rhs) : m_data(rhs.m_data) {}

    GTEntry_t& operator=(const GTEntry_t& rhs) {
      m_data = rhs.m_data;
      return *this;
    }

    const std::string& recordName() const { return std::get<0>(m_data); }
    const std::string& recordLabel() const { return std::get<1>(m_data); }
    const std::string& tagName() const { return std::get<2>(m_data); }
    std::size_t hashvalue() const {
      // Derived from CondDB v1 TagMetadata implementation.
      // Unique Keys constructed with Record and Labels - allowing for multiple references of the same Tag in a GT
      std::hash<std::string> hasher;
      std::string key = recordName();
      if (!recordLabel().empty())
        key = key + "_" + recordLabel();
      std::size_t result = hasher(key);
      return result;
    }
    bool operator<(const GTEntry_t& toCompare) const { return this->hashvalue() < toCompare.hashvalue(); }

  private:
    std::tuple<std::string, std::string, std::string> m_data;
  };

  struct RunInfo_t {
    RunInfo_t() : run(0), start(), end() {}
    RunInfo_t(const std::tuple<long long unsigned int, boost::posix_time::ptime, boost::posix_time::ptime>& data)
        : run(std::get<0>(data)), start(std::get<1>(data)), end(std::get<2>(data)) {}
    bool isOnGoing() {
      if (run == 0)
        return false;
      auto now = boost::posix_time::second_clock::universal_time();
      return (start < now) && (end == start);
    }
    Time_t run;
    boost::posix_time::ptime start;
    boost::posix_time::ptime end;
  };

}  // namespace cond

#endif
