#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"

#include <iostream>

using namespace dqm::me_util;
using namespace dqm::qtests;

using std::cout; using std::endl; using std::cerr;
using std::vector; using std::string; using std::set;


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

// (a) reset modifications to monitorable since last cycle 
// (b) reset sets of added/removed/updated contents and updated QReports
void DaqMonitorBEInterface::resetStuff(void)
{
  // reset added, removed monitorable
  addedMonitorable.clear();
  removedMonitorable.clear();
  // reset added, removed contents;
  addedContents.clear();
  removedContents.clear();
  // reset updated contents
  updatedContents.clear();
  // reset updated QReports
  updatedQReports.clear();

  resetWasCalled = true;
}

/* come here at end of monitoring cycle for all receivers;
   (a) call resetUpdate for modified contents
   
   (b) if resetMEs=true, reset MEs that were updated (and have resetMe = true);
   [flag resetMe is typically set by sources (false by default)];
   [Clients in standalone mode should also have resetMEs = true] 
   
   (c) if callResetStuff = true, call resetStuff
   (typical behaviour: Sources & Collector have callResetStuff = true, whereas
   clients have callResetStuff = false, so GUI/WebInterface can access the 
   modifications in monitorable & monitoring) */
void DaqMonitorBEInterface::doneSending(bool resetMEs, bool callResetStuff)
{

  // reset "update" flag for monitoring objects that have been updated/added
  for(monit_it path = updatedContents.begin(); 
      path != updatedContents.end(); ++path)
    { // loop over all pathnames/directories
      
      string pathname = path->first;
      
      for(sIt it = path->second.begin(); it != path->second.end(); ++it)
	{ // loop over all ME names
	  
	  MonitorElement * me = findObject(*it, pathname);
	  if(me)
	    {
	      // if reset, reset (ie. clear contents of) monitoring element
	      if(resetMEs && me->resetMe())me->Reset();
	      me->resetUpdate();
	    }
	  
	} // loop over all ME names
      
    }  // loop over all pathnames/directories
  
  // reset "update" flag for QReports that have been updated/added
  for(set<QReport *>::iterator it = updatedQReports.begin(); 
      it != updatedQReports.end(); ++it)
    {      
      if(*it)
	(*it)->resetUpdate();
    }
  
  // if flag=true, reset list of modified monitorable & monitoring
  if(callResetStuff)resetStuff();

}

// reset ME contents 
void DaqMonitorBEInterface::softReset(MonitorElement * me)
{
  if(!checkElement(me))
    return;

  me->softReset();
}

// reverts action of softReset
void DaqMonitorBEInterface::disableSoftReset(MonitorElement * me)
{
  if(!checkElement(me))
    return;

  me->disableSoftReset();
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
void DaqMonitorBEInterface::add2UpdatedContents(string name, string pathname)
{
  monit_it It = updatedContents.find(pathname);
  if(It == updatedContents.end())
    {
      set<string> temp; temp.insert(name);
      updatedContents[pathname] = temp;
    }
  else
    It->second.insert(name);
}

// run quality tests (also finds updated contents in last monitoring cycle,
// including newly added content) 
void DaqMonitorBEInterface::runQTests(void)
{
  if(!wasResetCalled())
    {
      cout << " *** Warning! Need to call MonitorUserInterface::doMonitoring\n"
	   << " before calling MonitorUserInterface::runQTests again! " << endl;
    }

  // keep track here of modified algorithm since last time runQTests ran
  vector<QCriterion *> modifiedAlgos;
  for(qc_it qc = qtests_.begin(); qc != qtests_.end(); ++qc)
    { // loop over quality tests
      if( (qc->second)->wasModified() )
	modifiedAlgos.push_back(qc->second);

    } // loop over quality tests

  try
    {      
      // first, check if qtests_ should be attached to any of the added elements
      checkAddedElements();

      // now run the quality tests for real!
      runQualityTests();
      
    } // try-block

  catch(ME_map * m)
    {
      nullError("ME_map");
    }
  catch(MonitorElement * m)
    {
      nullError("MonitorElement");
    }
  catch (...)
    {
      cerr << " *** Unknown error returned by DaqMonitorBEInterface::runQTests " 
	   << endl;
    }

  // reset "wasModified" flag for quality-test algorithms
  for(vqc_it it = modifiedAlgos.begin(); it != modifiedAlgos.end(); ++it)
    (*it)->wasModified_ = false;

  resetWasCalled = false;
}

// run quality tests (also finds updated contents in last monitoring cycle,
// including newly added content) <-- to be called only by runQTests
void DaqMonitorBEInterface::runQualityTests(void)
{
  for(cglob_it path = own.global_.begin(); path != own.global_.end(); 
      ++path)
    { // loop over all pathnames 
      
      if(!path->second)
	throw path->second;
      
      for(cME_it it = path->second->begin(); it != path->second->end(); 
	  ++it)
	{ // loop over monitoring objects in current folder
	  
	  // this is probably a ME that appears only on monitorable
	  if(!it->second) 
	    continue;
	  
	  if(it->second->wasUpdated())
	    add2UpdatedContents(it->first, path->first);

	  // quality tests should be run if (a) ME has been modified, or
	  // (b) algorithm has been modified; 
	  // this is done in MonitorElement::runQTests()
	  it->second->runQTests();
	  
	    
	} // loop over monitoring objects in current folder

    } // loop over all pathnames 

}

// get QCriterion corresponding to <qtname> 
// (null pointer if QCriterion does not exist)
QCriterion * DaqMonitorBEInterface::getQCriterion(string qtname) const
{
  cqc_it it = qtests_.find(qtname);
  if(it == qtests_.end())
    return (QCriterion *) 0;
  else
    return it->second;
}

// get QReport from ME (null pointer if no such QReport)
QReport * 
DaqMonitorBEInterface::getQReport(MonitorElement * me, string qtname)
{
  QReport * ret = 0;
  if(me)
    {
      qr_it it = me->qreports_.find(qtname);
      if(it != me->qreports_.end())
	ret = it->second;
    }

  return ret;
}


// attach quality test <qtname> to all ME matching <search_string>;
// <search_string> could : (a) be exact pathname (e.g. A/B/C/histo)
// (b) include wildcards (e.g. A/?/C/histo, A/B/*/histo or A/B/*);
// this action applies to all MEs already available or future ones
void DaqMonitorBEInterface::useQTest(string search_string, string qtname) const
{
  QCriterion * qc = getQCriterion(qtname);
  if(!qc)
    {
      cerr << " *** Quality test " << qtname << " does not exist! " << endl;
      return;
    }

  qc->searchStrings.insert(search_string);
  scanContents(qc, search_string);
}

// look for all MEs matching <search_string> in own.global_;
// if found, create QReport from QCriterion and add to ME
void DaqMonitorBEInterface::scanContents(QCriterion * qc, const string & 
					 search_string) const
{
  cglob_it start, end, parent_dir;
  getSubRange<global_map>(search_string,own.global_,start,end, parent_dir);

  for(cglob_it path = start; path != end; ++path)
    // loop over pathnames of global_map subrange
    scanContents(qc, search_string, path);

  if(parent_dir != own.global_.end())
    scanContents(qc, search_string, parent_dir);
}

// same as scanContents above but for one path only
void DaqMonitorBEInterface::scanContents(QCriterion * qc, 
					 const string & search_string,
					 cglob_it & path) const
{
  string pathname = path->first;
  for(cME_it file = path->second->begin(); file != path->second->end(); 
      ++file)
    { // loop over files of <pathname>
      
      string fullname = getUnixName(pathname, file->first);
      
      if(matchit(fullname, search_string))
	{ // this is a match!
	  MonitorElement * me = (MonitorElement *)file->second;
	  if(me)
	    addQReport(me, qc);
	  
	} // this is a match!
	  
    } // loop over files of <pathname>

}

// loop over quality tests & addedContents: look for MEs that 
// match QCriterion::searchStrings; upon a match, add QReport to ME(s)
void DaqMonitorBEInterface::checkAddedElements(void)
{
  for(cqc_it qc = qtests_.begin(); qc != qtests_.end(); ++qc)
    { // loop over quality tests
      
      for(csIt search_string = (qc->second)->searchStrings.begin(); 
	  search_string != (qc->second)->searchStrings.end(); 
	  ++search_string)
	// loop over search-strings for quality test
	checkAddedElements(*search_string, qc);

    } // loop over quality tests
}

// loop over addedContents: look for MEs that 
// match search_string; upon a match, add QReport to ME(s)
void DaqMonitorBEInterface::checkAddedElements(const string & search_string,
					       cqc_it & qc)
{  
  cmonit_it start, end, parent_dir;
  getSubRange<monit_map>(search_string, addedContents, start, end, 
			 parent_dir);
  
  for(cmonit_it path = start; path != end; ++path)
    // loop over specified range of added contents
    checkAddedElements(search_string, qc, path);

  if(parent_dir != addedContents.end())
    checkAddedElements(search_string, qc, parent_dir);

}

// same as checkAddedElements above for only one path
void DaqMonitorBEInterface::checkAddedElements(const string & search_string, 
					       cqc_it & qc,
					       cmonit_it & path)
{
  for(csIt it = path->second.begin(); it!= path->second.end(); ++it)
    { // loop over all added MEs
      
      // get unix-like filename
      string fullname = getUnixName(path->first, *it);
      if(matchit(fullname, search_string))
	{
	  // this is a match!
	  MonitorElement* me = findObject(*it, path->first);
	  /* I need to double-check that qreport is not already added to "me";
	     This is because there is a chance that users may
	     1. define a ME after resetStuff has been called
	     2. call MonitorUserInterface::useQTest
	     3. and then call MonitorUserInterface::runQTests, which
	     eventually calls this function
	     In this case ME appears in addedContents and this call would
	     give an error... (not sure of a better way right now)
	  */
	  if(me && !me->getQReport(qc->first))
	    addQReport(me, qc->second);
	      
	} // this is a match!

    } // loop over all added MEs
      
}

DaqMonitorBEInterface::~DaqMonitorBEInterface(void)
{
  // remove all quality tests
  for(qc_it it = qtests_.begin(); it != qtests_.end(); ++it)
    {
      if(it->second)
	delete it->second;
    }
  qtests_.clear();

  if(dqm_locker)delete dqm_locker;
}


// acquire lock
void DaqMonitorBEInterface::lock()
{
  // cout << " Called lock " << endl;
  // mutex is not released till previous lock has been deleted in unlock()
  dqm_locker = 
    new boost::mutex::scoped_lock(*edm::rootfix::getGlobalMutex());
}

// release lock
void DaqMonitorBEInterface::unlock()
{
  //  cout << " Called unlock " << endl;
  if(dqm_locker)
    {
      // use local stack pointer to release memory, so we do not set
      // dqm_locker to zero AFTER lock has been released
      // (dangerous, as another thread may have acquired lock in the meantime)
      boost::mutex::scoped_lock * tmp_lock = dqm_locker;
      dqm_locker = 0;
      delete tmp_lock;
    }
  //  
}


const string DaqMonitorBEInterface::monitorDirName = "DQMData";
const string DaqMonitorBEInterface::subscriberDirName = "Subscribers";
