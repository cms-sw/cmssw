#ifndef RUNSUMMARY_HANDLER_H
#define RUNSUMMARY_HANDLER_H

#include <string>

#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondFormats/RunInfo/interface/RunSummary.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

class RunSummaryHandler : public popcon::PopConSourceHandler<RunSummary> {
public:
  void getNewObjects() override;
  std::string id() const override { return m_name; }
  ~RunSummaryHandler() override;
  RunSummaryHandler(const edm::ParameterSet& pset);

private:
  std::string m_name;
  unsigned long long m_since;

  // for reading from omds

  std::string m_connectionString;

  std::string m_authpath;
  std::string m_host;
  std::string m_sid;
  std::string m_user;
  std::string m_pass;
  int m_port;
};

#endif
