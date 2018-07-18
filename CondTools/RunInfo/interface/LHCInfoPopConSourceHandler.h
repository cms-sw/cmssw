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
  bool getNextFillData( cond::persistency::Session& session, const boost::posix_time::ptime& targetTime, bool ended );
  bool getFillData( cond::persistency::Session& session, unsigned short fillId );
  size_t getLumiData( cond::persistency::Session& session, const boost::posix_time::ptime& beginFillTime, const boost::posix_time::ptime& endFillTime );
  bool getDipData( cond::persistency::Session& session, const boost::posix_time::ptime& beginFillTime, const boost::posix_time::ptime& endFillTime );
  bool getCTTPSData( cond::persistency::Session& session, const boost::posix_time::ptime& beginFillTime, const boost::posix_time::ptime& endFillTime );
  bool getEcalData(  cond::persistency::Session& session, const boost::posix_time::ptime& lowerTime, const boost::posix_time::ptime& upperTime, bool update );

 private:
  bool m_debug;
  // starting date for sampling
  boost::posix_time::ptime m_startTime;
  boost::posix_time::ptime m_endTime;
  // sampling interval in seconds
  unsigned int m_samplingInterval;
  bool m_endFill = true;
  std::string m_name;  
  //for reading from relational database source 
  std::string m_connectionString, m_ecalConnectionString;
  std::string m_dipSchema, m_authpath;
  std::unique_ptr<LHCInfo> m_fillPayload;
  std::shared_ptr<LHCInfo> m_prevPayload;
  std::vector<std::pair<cond::Time_t,std::shared_ptr<LHCInfo> > > m_tmpBuffer;
  std::vector<std::shared_ptr<LHCInfo> > m_payloadBuffer;
  bool m_lastPayloadEmpty = false;
 };
  
#endif
