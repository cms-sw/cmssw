#ifndef RUNINFO_HANDLER_H
#define RUNINFO_HANDLER_H

#include <string>

#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondFormats/RunInfo/interface/RunInfo.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

class RunInfoHandler : public popcon::PopConSourceHandler<RunInfo>{
 public:
  void getNewObjects();
  std::string id() const { return m_name; }
  ~RunInfoHandler();
  RunInfoHandler(const edm::ParameterSet& pset); 
  
 private:
  std::string m_name;
  unsigned long long m_since;
  
  // for reading from omds
  std::string m_connectionString;
  std::string m_authpath;
  std::string m_user;
  std::string m_pass;
};

#endif 
