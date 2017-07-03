#ifndef L1TRIGGERSCALER_HANDLER_H
#define L1TRIGGERSCALER_HANDLER_H

#include <string>
#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondFormats/RunInfo/interface/L1TriggerScaler.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

class L1TriggerScalerHandler : public popcon::PopConSourceHandler<L1TriggerScaler>{
 public:
  void getNewObjects() override;
  std::string id() const override { return m_name;}
  ~L1TriggerScalerHandler() override;
  L1TriggerScalerHandler(const edm::ParameterSet& pset); 
 private:
  std::string m_name;
  unsigned long long m_since;
  
  // for reading from omds 
  
  std::string  m_connectionString;
  
  std::string m_authpath;
  std::string m_host;
  std::string m_sid;
  std::string m_user;
  std::string m_pass;
  int m_port;
};

#endif 
