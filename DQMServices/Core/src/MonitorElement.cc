#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/QCriterion.h"
#include "DQMServices/Core/interface/QTestStatus.h"

#include <iostream>

using namespace dqm::qtests;

using std::cout; using std::endl; using std::cerr;
using std::string;

MonitorElement::MonitorElement() 
{
  accumulate_on = false;
  qreports_.clear(); 
}

MonitorElement::MonitorElement(const char*name) 
{
  accumulate_on = false;
  qreports_.clear(); 
}

MonitorElement::~MonitorElement() 
{
  for(qr_it it = qreports_.begin(); it != qreports_.end(); ++it)
    {
      if(it->second)
	delete it->second;
    }

  qreports_.clear();
}

bool MonitorElement::wasUpdated() const {return man.variedSince;}


void MonitorElement::update() {man.variedSince = true;}

void MonitorElement::resetUpdate() {man.variedSince = false;}

bool MonitorElement::isFolder(void) const {return man.folder_flag;}
bool MonitorElement::isNotFolder(void) const {return !isFolder();}

// true if ME should be reset at end of monitoring cycle
bool MonitorElement::resetMe(void) const{return man.resetMe;}

// set resetMe flag (default: false)
void MonitorElement::setResetMe(bool flag)
{
  cout << " \"resetMe\" flag for monitoring element " << getName() 
       << " set to";
  if(flag)
    cout << " true";
  else
    cout << " false";
  cout << endl;

  man.resetMe = flag;
}

// if true, will accumulate ME contents (over many periods)
// until method is called with flag = false again
void MonitorElement::setAccumulate(bool flag)
{
  accumulate_on = flag;

  cout << " \"accumulate\" option has been";
  if(accumulate_on)
    cout << " en";
  else
    cout << " dis";
  cout << "abled for " << getName() << endl;  
}

// add quality report (to be called by DaqMonitorROOTBackEnd)
void MonitorElement::addQReport(QReport * qr)
{
  if(!qr)
    {
      cerr << " *** Cannot add null QReport to  MonitorElement " 
	   << getName() << endl;
      return;
    }

  string qtname = qr->getQRName();
  if(qreportExists(qtname))
    {
      delete qr; qr = 0;
      return;
    }

  qreports_[qtname] = qr;
  qr->setMonitorElement(this);
}

// get QReport corresponding to <qtname> (null pointer if QReport does not exist)
const QReport * MonitorElement::getQReport(string qtname) const
{
  cqr_it it = qreports_.find(qtname);
  if(it == qreports_.end())
    return (QReport *) 0;
  else
    return it->second;
}

// true if QReport with name <qtname> already exists
bool MonitorElement::qreportExists(string qtname) const
{
  if(getQReport(qtname))
   {
     cerr << " *** Quality report " << qtname 
	  << " already exists for MonitorElement " << getName() << endl;
     return true;
   }
  return false;
}

// for folders: get pathname;
// (class method to be overwritten for derived class)
//  for other MEs: get pathname of parent folder
string MonitorElement::getPathname() const
{
  string pathname;
  MonitorElement * parent = (MonitorElement *) parent_;
  if(parent)pathname = parent->getPathname();
  return pathname;
}

 
// run all quality tests
void MonitorElement::runQTests(void)
{
  qwarnings_.clear(); qerrors_.clear(); qothers_.clear();
  for(qr_it it = qreports_.begin(); it != qreports_.end(); ++it)
    { // loop over all quality reports/tests for MonitorElement
      QReport * qr = it->second;
      if(!qr)
	{
	  cerr << " *** Attempt to access null QReport " << it->first 
	       << " for MonitorElement " << getName() << endl;
	  continue;
	}
     
      // test should run if (a) ME has been modified, 
      // or (b) algorithm has been modified
      if(wasUpdated() || qr->getQCriterion()->wasModified() )
	qr->runTest();

      int status = qr->getStatus();
      switch(status)
	{
	case dqm::qstatus::WARNING:
	  qwarnings_.push_back(qr);
	  break;
	case dqm::qstatus::ERROR:
	  qerrors_.push_back(qr);
	  break;
	case dqm::qstatus::STATUS_OK:
	  break;
	default:
	  // all other cases go here
	  qothers_.push_back(qr);
	}

    } // loop over QReports

}
