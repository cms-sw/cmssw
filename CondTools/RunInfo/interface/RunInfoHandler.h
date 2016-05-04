#ifndef RUNINFO_HANDLER_H
#define RUNINFO_HANDLER_H

#include <string>

#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondFormats/RunInfo/interface/RunInfo.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class RunInfoHandler : public popcon::PopConSourceHandler<RunInfo>{
 public:
  void getNewObjects();
  std::string id() const { return m_name; }
  ~RunInfoHandler();
  RunInfoHandler(const edm::ParameterSet& pset); 
  
 private:
  unsigned long long m_since;
  std::string m_name;
  
  // for reading from omds
  std::string m_runinfo_schema;
  std::string m_dcsenv_schema;
  std::string m_connectionString;
  edm::ParameterSet m_connectionPset;
};

#endif 
