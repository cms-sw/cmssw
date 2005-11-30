#include "DQMServices/Core/interface/QTestStatus.h"
#include "DQMServices/Core/interface/QCriterion.h"

#include <iostream>
#include <sstream>

using namespace std;

const float QCriterion::ERROR_PROB_THRESHOLD = 0.50;
const float QCriterion::WARNING_PROB_THRESHOLD = 0.90;

// initialize values
void QCriterion::init(void)
{
  enabled_ = wasModified_ = true;
  errorProb_ = ERROR_PROB_THRESHOLD; 
  warningProb_ = WARNING_PROB_THRESHOLD;
  setAlgoName("NO_ALGORITHM");
  status_ = dqm::qstatus::DID_NOT_RUN;
  message_ = "NO_MESSAGE";
}

// make sure algorithm can run (false: should not run)
bool QCriterion::check(const MonitorElement * const me)
{
  // reset result
  prob_ = -1;

  if(!isEnabled())
    {
      setDisabled();
      return false;
    }

  if(!me)
    {
      setInvalid();
      return false;
    }

  if(notEnoughStats(me))
    {
      setNotEnoughStats();
      return false;
    }

  // if here, we can run the test
  return true;
}

// set status & message for disabled tests
void QCriterion::setDisabled(void)
{
  status_ = dqm::qstatus::DISABLED;
  std::ostringstream message;
  message << " Test " << qtname_ << " (" << getAlgoName() 
	  << ") has been disabled ";
  message_ = message.str();  
}

// set status & message for invalid tests
void QCriterion::setInvalid(void)
{
  status_ = dqm::qstatus::INVALID;
  std::ostringstream message;
  message << " Test " << qtname_ << " (" << getAlgoName() 
	  << ") cannot run due to problems ";
  message_ = message.str();  
}

// set status & message for tests w/o enough statistics
void QCriterion::setNotEnoughStats(void)
{
  status_ = dqm::qstatus::INSUF_STAT;
  std::ostringstream message;
  message << " Test " << qtname_ << " (" << getAlgoName() 
	  << ") cannot run (insufficient statistics) ";
  message_ = message.str();  
}

// set status & message for succesfull tests
void QCriterion::setOk(void)
{
  status_ = dqm::qstatus::STATUS_OK;
  setMessage();
}

// set status & message for tests w/ warnings
void QCriterion::setWarning(void)
{
  status_ = dqm::qstatus::WARNING;
  setMessage();
}

// set status & message for tests w/ errors
void QCriterion::setError(void)
{
  status_ = dqm::qstatus::ERROR;
  setMessage();
}
