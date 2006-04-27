#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"

#include <iostream>

using namespace std;
using namespace dqm::me_util;
using namespace dqm::qtests;

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
}

/* come here at end of monitoring cycle for all receivers;
   (a) call resetUpdate for modified contents
   
   (b) if resetMEs=true, reset MEs that were updated (and have resetMe = true);
   [flag resetMe is typically set by sources (false by default)];
   
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
  for(meIt it = updatedQReports.begin(); it != updatedQReports.end(); ++it)
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
void DaqMonitorBEInterface::add2UpdatedContents(const string & name, 
						const string & pathname)
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
void DaqMonitorBEInterface::scanContents(QCriterion * qc, string search_string) 
  const
{
  for(cglob_it path = own.global_.begin(); path != own.global_.end(); 
      ++path)
    { // loop over pathnames of global_map
      
      string pathname = path->first;
      for(cME_it file = path->second->begin(); 
	  file != path->second->end(); ++file)
	{ // loop over files of <pathname>
	  
	  string fullname = getUnixName(pathname, file->first);
	  
	  if(matchit(fullname, search_string))
	    { // this is a match!
	      MonitorElement * me = (MonitorElement *)file->second;
	      if(me)
		addQReport(me, qc);

	    } // this is a match!
	  
	} // loop over files of <pathname>
      
    } // loop over pathnames of global_map

}

// loop over addedContents, look for MEs that match QCriterion::searchStrings 
// (by looping over all quality tests); upon a match, add QReport to ME(s)
void DaqMonitorBEInterface::checkAddedElements(void)
{

  if(qtests_.empty())return;
  
  for(cmonit_it path = addedContents.begin(); path != addedContents.end(); 
      ++path)
    {
      // loop over all pathnames of added contents
      
      for(csIt it = path->second.begin(); it!= path->second.end(); ++it)
	// loop over all added MEs
	checkAddedElement(path->first, *it);
      
    } // loop over all pathnames of added contents
    
}

// check if ME matches any of QCriterion::searchStrings;
// upon a match, add QReport to ME(s)
void DaqMonitorBEInterface::checkAddedElement(string pathname, string ME_name)
{  
  for(cqc_it qc = qtests_.begin(); qc != qtests_.end(); ++qc)
    { // loop over quality tests
      
      for(csIt search_string = (qc->second)->searchStrings.begin(); 
	  search_string != (qc->second)->searchStrings.end(); 
	  ++search_string)
	{ // loop over search-strings for quality test
      
	  // get unix-like filename
	  string fullname = getUnixName(pathname, ME_name);
	  if(matchit(fullname, *search_string))
	    {
	      // this is a match!
	      MonitorElement* me = findObject(ME_name, pathname);
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

	} // loop over search-strings for quality tests
      
    } // loop over quality tests

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
}


const string DaqMonitorBEInterface::monitorDirName = "DQMData";
const string DaqMonitorBEInterface::subscriberDirName = "Subscribers";
