#ifndef DQMSERVICES_CORE_Q_REPORT_H
# define DQMSERVICES_CORE_Q_REPORT_H

# include "DQMServices/Core/interface/DQMDefinitions.h"
# include "DQMServices/Core/interface/DQMNet.h"
# include <vector>
# include <string>

class QCriterion;

/** Class for reporting results of quality tests for Monitoring Elements */
class QReport
{
public:
  /// get test status (see Core/interface/QTestStatus.h)
  int getStatus(void) const
    { return qvalue_->code; }

  /// get test result i.e. prob value
  float getQTresult(void) const
    { return qvalue_->qtresult; }

  /// get message attached to test
  const std::string &getMessage(void) const
    { return qvalue_->message; }

  /// get name of quality test
  const std::string &getQRName(void) const
    { return qvalue_->qtname; }

  /// get vector of channels that failed test
  /// (not relevant for all quality tests!)
  const std::vector<DQMChannel> &getBadChannels(void) const
    { return badChannels_; }

  /// get QCriterion
  const QCriterion *getQCriterion(void) const
    { return qcriterion_; }

private:
  friend class QCriterion;
  friend class MonitorElement;  // for running the quality test
  friend class DQMStore;        // for setting QReport parameters after receiving report

  QReport(DQMNet::QValue *value, QCriterion *qc)
    : qvalue_ (value),
      qcriterion_ (qc)
    {}

  DQMNet::QValue	  *qvalue_;	//< Pointer to the actual data.
  QCriterion		  *qcriterion_;	//< Pointer to QCriterion algorithm.
  std::vector<DQMChannel> badChannels_; //< Bad channels from QCriterion.
}; 

#endif // DQMSERVICES_CORE_Q_REPORT_H
