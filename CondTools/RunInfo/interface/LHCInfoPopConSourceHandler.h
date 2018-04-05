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
  bool m_debug;
  unsigned short m_firstFill, m_lastFill;
  std::string m_name;  
  //for reading from relational database source 
  std::string m_connectionString, m_dipSchema, m_authpath;
  };
  
#endif
