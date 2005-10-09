#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"

#include <iostream>

using namespace std;
using namespace dqm::me_util;

// check against null objects
bool DaqMonitorBEInterface::checkElement(const MonitorElement * const me) const
{
  if(!me)
    {
      cerr << " *** Error! Null monitoring element " << endl;
      return false;
    }
  return true;
}

// check if object is really a folder
bool DaqMonitorBEInterface::checkFolder(const MonitorElement * const dir) const
{
  if(!checkElement(dir))
    return false;

  if(!dir->isFolder())
    {
      cerr <<" *** Error! " << dir->getName() << " is not a folder " << endl;
      return false;
    }

  return true;
}

// get added contents (since last cycle)
// return vector<string> of the form: <dir pathname>:<obj1>,<obj2>,...
void DaqMonitorBEInterface::getAddedContents(vector<string> & put_here) const
{
  convert(put_here, addedContents);
}

// get removed contents (since last cycle)
// return vector<string> of the form: <dir pathname>:<obj1>,<obj2>,...
void DaqMonitorBEInterface::getRemovedContents(vector<string> & put_here) const
{
  convert(put_here, removedContents);
}

// get updated contents (since last cycle)
// COMPLEMENTARY to addedContents, removedContents
// return vector<string> of the form: <dir pathname>:<obj1>,<obj2>,...
void DaqMonitorBEInterface::getUpdatedContents(vector<string> & put_here) const
{
  convert(put_here, updatedContents);
}

// convert dqm::me_util::monit_map into 
// vector<string> of the form: <dir pathname>:<obj1>,<obj2>,...
// to be invoked by getAddedContents, getRemovedContents, getUpdatedContents
void 
DaqMonitorBEInterface::convert(vector<string> & put_here, const monit_map & in) 
  const
{
  put_here.clear();
  for(cmonit_it path = in.begin(); path != in.end(); ++path)
    { // loop over all pathnames/directories
      string new_entry;
      const set<string> & input = path->second;
      
      for(csIt it = input.begin(); it != input.end(); ++it)
	{ // loop over all monitoring elements names

	  if(it != input.begin())
	    new_entry += ",";
	  new_entry += *it;
	} // loop over all monitoring elements names

      if(!new_entry.empty())
	{ // directory contains at least one monitoring element
	  new_entry = path->first + ":" + new_entry;
	  put_here.push_back(new_entry);
	}

    } // loop over all pathnames/directories
}

// get added monitorable (since last cycle)
// return vector<string> of the form: <dir pathname>:<obj1>,<obj2>,...
void DaqMonitorBEInterface::getAddedMonitorable(vector<string> & put_here) const
{
  put_here = addedMonitorable;
}

// get removed monitorable (since last cycle)
// return vector<string> of the form: <dir pathname>:<obj1>,<obj2>,...
void DaqMonitorBEInterface::getRemovedMonitorable(vector<string> & put_here) const
{
  put_here = removedMonitorable;
}

// reset modifications to monitorable since last cycle
void DaqMonitorBEInterface::resetNewMonitorable(void)
{
  // reset added, removed monitorable
  addedMonitorable.clear();
  removedMonitorable.clear();
  // reset added, removed contents;
  addedContents.clear();
  removedContents.clear();
  // reset updated contents
  updatedContents.clear();
}

// come here at end of monitoring cycle for all receivers;
// if reset=true, reset MEs that were updated (and have resetMe = true)
void DaqMonitorBEInterface::doneSending(bool reset)
{
  resetNewMonitorable();

  // reset "update" flag for monitoring objects that have been (updated, and) sent
  for(set<MonitorElement *>::iterator it = collectiveMonitoring.begin(); 
      it != collectiveMonitoring.end(); ++it)
    {
      // if reset, reset (ie. clear contents of) monitoring element
      if(reset && (*it)->resetMe())(*it)->Reset();

      (*it)->resetUpdate();
    }

  collectiveMonitoring.clear();
}

// reset ME contents 
void DaqMonitorBEInterface::softReset(MonitorElement * me)
{
  if(!checkElement(me))
    return;

  me->softReset();
}

// if true: will subtract contents copied at "soft-reset" from now on
// if false: will NO longer subtract contents (default)
void DaqMonitorBEInterface::enableSoftReset(MonitorElement * me, bool flag)
{
  if(!checkElement(me))
    return;

  me->enableSoftReset(flag);
}

// if true, will accumulate ME contents (over many periods)
// until method is called with flag = false again
void DaqMonitorBEInterface::setAccumulate(MonitorElement * me, bool flag)
{
  if(!checkElement(me))
    return;

  me->setAccumulate(flag);
}

// add <name> to back-end interface's updatedContents
void DaqMonitorBEInterface::add2UpdatedContents(const string & name)
{
  string pathname = pwd();
  monit_it It = updatedContents.find(pathname);
  if(It == updatedContents.end())
    {
      set<string> temp; temp.insert(name);
      updatedContents[pathname] = temp;
    }
  else
    It->second.insert(name);
}

const string DaqMonitorBEInterface::monitorDirName = "DQMData";
const string DaqMonitorBEInterface::subscriberDirName = "Subscribers";
