#ifndef LHCINFOPOPCONSOURCEHANDLER_H
#define LHCINFOPOPCONSOURCEHANDLER_H

#include <string>

#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondFormats/RunInfo/interface/LHCInfo.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

class LHCInfoPopConSourceHandler : public popcon::PopConSourceHandler<LHCInfo>{
 public:
  LHCInfoPopConSourceHandler( const edm::ParameterSet& pset ); 
  ~LHCInfoPopConSourceHandler() override;
  void getNewObjects() override;
  std::string id() const override;
private:
 void addEmptyPayload( cond::Time_t iov );
 void addPayload( LHCInfo& newPayload, cond::Time_t iov );
 bool getFillData( cond::persistency::Session& session, const boost::posix_time::ptime& targetTime, bool next, LHCInfo& payload );
 bool getCurrentFillData( cond::persistency::Session& session, const boost::posix_time::ptime& targetTime, LHCInfo& payload );
 bool getNextFillData( cond::persistency::Session& session, const boost::posix_time::ptime& targetTime, LHCInfo& payload );
 bool getLumiData( cond::persistency::Session& session, const boost::posix_time::ptime& targetTime, LHCInfo& payload );
 bool getDipData( cond::persistency::Session& session, const boost::posix_time::ptime& targetTime, LHCInfo& payload );
 bool getCTTPSData( cond::persistency::Session& session, const boost::posix_time::ptime& targetTime, LHCInfo& payload );
 bool getEcalData(  cond::persistency::Session& session, const boost::posix_time::ptime& targetTime, LHCInfo& payload );

 private:
  bool m_debug;
  // starting date for sampling
  boost::posix_time::ptime m_startTime;
  boost::posix_time::ptime m_endTime;
  // sampling interval in seconds
  unsigned int m_samplingInterval;
  std::string m_name;  
  //for reading from relational database source 
  std::string m_connectionString, m_ecalConnectionString;
  std::string m_dipSchema, m_authpath;
  std::vector<std::unique_ptr<LHCInfo> > m_payloadBuffer;
  bool m_lastPayloadEmpty = false;
 };
  
#endif
