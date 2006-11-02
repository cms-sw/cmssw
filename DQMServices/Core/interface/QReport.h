#ifndef _QREPORT_H_
#define _QREPORT_H_

#include <string>
#include <map>

#include "DQMServices/Core/interface/DQMDefinitions.h"

class QCriterion;
class MonitorElement;

/** Class for reporting results of quality tests for Monitoring Elements */
class QReport
{
 public:
  /// get test status (see Core/interface/QTestStatus.h)
  int getStatus(void) const {return status_;}
  /// get message attached to test
  std::string getMessage(void) const {return message_;} 
  /// get name of quality test
  std::string getQRName(void) const {return qtname_;}

  /// get vector of channels that failed test
  /// (not relevant for all quality tests!)
  std::vector<dqm::me_util::Channel> getBadChannels(void) const
  {return badChannels_;}

  static std::string getNullMessage(void) {return "NULL_MESSAGE";}

 protected:
  /// to be used when unpacking quality reports (ie. in ReceiverBase class)
  QReport(std::string qtname);
  /// to be used by DaqMonitorROOTBackEnd class 
  /// (when QCriterion added to MonitorElement object)
  QReport(QCriterion * qc);
  ///
  virtual ~QReport(void) {}
  /// get QCriterion
  const QCriterion * getQCriterion(void) const {return qcriterion_;}
  /// run QCriterion algorithm
  void runTest(void);
  /// set quality test name
  inline void setName(std::string qtname){qtname_ = qtname;}
  /// reset status & message
  void resetStatusMessage(void);
  /// set status (to be called by ReceiverBase)
  inline void setStatus(int status){status_ = status;}
  /// set message (to be called by ReceiverBase)
  inline void setMessage(std::string message){message_ = message;}
  /// initialization
  void init(void);

  /// to be called after test has run and status/message have been updated
  /// (implemented in derived class)
  virtual void updateReport(void) = 0;
  /// to be called after QReport has been sent downstream
  /// (implemented in derived class)
  virtual void resetUpdate(void) = 0;
  
  //
  std::vector<dqm::me_util::Channel> badChannels_;

 private:
  /// pointer to QCriterion algorithm
  QCriterion * qcriterion_;
  /// quality test status (see Core/interface/QTestStatus.h)
  int status_;
  /// message attached to test
  std::string message_;
  /// quality test name
  std::string qtname_;

  /// this is the ME for which QReport is about
  MonitorElement * myME_;
  /// set myME
  void setMonitorElement(MonitorElement * me){myME_ = me;}
  /// for running the quality test
  friend class MonitorElement;
  /// for setting QReport parameters after receiving report
  friend class ReceiverBase;
  /// for calling resetUpdate
  friend class DaqMonitorBEInterface;
}; 

namespace dqm
{
  namespace qtests
  {
    /// key: quality test name, value: address of QReport
    typedef std::map<std::string, QReport *> QR_map;
    typedef QR_map::iterator qr_it;
    typedef QR_map::const_iterator cqr_it;
  }
}

#endif
