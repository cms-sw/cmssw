#ifndef DQMSERVICES_CORE_Q_REPORT_H
#define DQMSERVICES_CORE_Q_REPORT_H

#include "DQMServices/Core/interface/DQMDefinitions.h"
#include "DQMServices/Core/interface/DQMNet.h"
#include <vector>
#include <string>

class QCriterion;
namespace dqm::impl {
  class MonitorElement;
  class DQMStore;
}  // namespace dqm::impl

/** Class for reporting results of quality tests for Monitoring Elements */
class QReport {
public:
  /// get test status (see Core/interface/QTestStatus.h)
  int getStatus() const { return qvalue_->code; }

  /// get test result i.e. prob value
  float getQTresult() const { return qvalue_->qtresult; }

  /// get message attached to test
  const std::string &getMessage() const { return qvalue_->message; }

  /// get name of quality test
  const std::string &getQRName() const { return qvalue_->qtname; }

  /// get vector of channels that failed test
  /// (not relevant for all quality tests!)
  const std::vector<DQMChannel> &getBadChannels() const { return badChannels_; }

private:
  friend class QCriterion;
  friend class dqm::impl::MonitorElement;  // for running the quality test
  friend class dqm::impl::DQMStore;        // for setting QReport parameters after receiving report

  QReport(DQMNet::QValue *value) : qvalue_(value) {}

  DQMNet::QValue *qvalue_;               //< Pointer to the actual data.
  std::vector<DQMChannel> badChannels_;  //< Bad channels from QCriterion.
};

#endif  // DQMSERVICES_CORE_Q_REPORT_H
