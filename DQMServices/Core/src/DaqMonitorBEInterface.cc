#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Core/interface/CollateMonitorElement.h"

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
// and sets of added/removed contents
void DaqMonitorBEInterface::resetMonitorableDiff()
{
  // reset added, removed monitorable
  addedMonitorable.clear();
  removedMonitorable.clear();
  // reset added, removed contents;
  addedContents.clear();
  removedContents.clear();
  // reset modified tags
  addedTags.clear();
  removedTags.clear();

  rMonitorableDiffWasCalled = true;
}

// reset updated contents and updated QReports
void DaqMonitorBEInterface::resetMonitoringDiff()
{
  // reset updated contents
  updatedContents.clear();
  // reset updated QReports
  updatedQReports.clear();

  rMonitoringDiffWasCalled = true;
}

/* come here after sending monitorable to all receivers;
   if callResetDiff = true, call resetMonitorableDiff
   (typical behaviour: Sources & Collector have callResetDiff = true, whereas
   clients have callResetDiff = false, so GUI/WebInterface can access the 
   modifications in monitorable & monitoring) */
void 
DaqMonitorBEInterface::doneSendingMonitorable(bool callResetDiff)
{
  // if flag=true, reset list of modified monitoring
  if(callResetDiff)resetMonitorableDiff();
}

/* come here after sending monitoring to all receivers;
   (a) call resetUpdate for modified contents:
   
   if resetMEs=true, reset MEs that were updated (and have resetMe = true);
   [flag resetMe is typically set by sources (false by default)];
   [Clients in standalone mode should also have resetMEs = true] 
   
   (b) if callResetDiff = true, call resetMonitoringDiff
   (typical behaviour: Sources & Collector have callResetDiff = true, whereas
   clients have callResetDiff = false, so GUI/WebInterface can access the 
   modifications in monitorable & monitoring) */
void 
DaqMonitorBEInterface::doneSendingMonitoring(bool resetMEs, bool callResetDiff)
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
  
  // if flag=true, reset list of modified monitoring
  if(callResetDiff)resetMonitoringDiff();

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
      if(!addedContents.empty())
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

  rMonitoringDiffWasCalled = rMonitorableDiffWasCalled = false;
}

// loop over quality tests & addedContents: look for MEs that 
// match QCriterion::rules; upon a match, add QReport to ME(s)
void DaqMonitorBEInterface::checkAddedElements(void)
{
  for(cqc_it qc = qtests_.begin(); qc != qtests_.end(); ++qc)
    { // loop over quality tests
      vME allMEs;
      checkAddedContents(qc->second->rules, allMEs);
      addQReport(allMEs, qc->second);
    } // loop over quality tests
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

// scan structure <rDir>, looking for all MEs matching <search_string>;
// put results in <put_here>
void DaqMonitorBEInterface::scanContents
(const string & search_string, const rootDir & rDir,  
 vector<MonitorElement *> & put_here) const
{
  put_here.clear();
  
  if(!hasWildCards(search_string))
    {
      MonitorElement * me = get(search_string);
      if(me)
	put_here.push_back(me);
    }
  else
    {

      cdir_it start, end, parent_dir;
      getSubRange<dir_map>(search_string, rDir.paths,start,end,parent_dir);
      
      // do parent directory first
      if(parent_dir != Own.paths.end())
	scanContents(search_string, parent_dir->second, put_here);
      
      for(cdir_it path = start; path != end; ++path)
	// loop over pathnames in directory structure
	scanContents(search_string, path->second, put_here);
    }
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

// get rootDir corresponding to tag 
// (Own for tag=0, or null for non-existing tag)
const rootDir * DaqMonitorBEInterface::getRootDir(unsigned int tag) const
{
  const rootDir * ret = 0;
  if(tag)
    {
      ctdir_it tg = Tags.find(tag);
      if(tg != Tags.end())
	ret = &(tg->second);
    }
  else // this corresponds to Own
    ret = &Own;

  return ret;
}

// check if added contents match rules; put matches in put_here
void DaqMonitorBEInterface::checkAddedContents
(const searchCriteria & rules, vector<MonitorElement *> & put_here) const
{
  for(csMapIt sc = rules.search.begin(); sc != rules.search.end(); ++sc)
    {
      const rootDir * Dir = getRootDir(sc->first);
      if(!Dir)continue;
      
      checkAddedSearchPaths(sc->second.search_path, *Dir, put_here);
      checkAddedFolders(sc->second.folders, *Dir, false, put_here);
      checkAddedFolders(sc->second.foldersFull, *Dir, true, put_here);
    }
  
  
  for(vector<unsigned int>::const_iterator tg = rules.tags.begin(); 
      tg != rules.tags.end(); ++tg)
    {
      const rootDir * Dir = getRootDir(*tg);
      if(!Dir)continue;
      checkAddedTags(*Dir, put_here);
    }
}



// check if added contents belong to folders 
// (use flag to specify if subfolders should be included)
void DaqMonitorBEInterface::checkAddedFolders
(const vector<string> & folders, const rootDir & Dir, bool useSubfolders,
 vector<MonitorElement*>& put_here) const
{
  for(cvIt f = folders.begin(); f != folders.end(); ++f)
    { // loop over folders to be watched
      
      if(useSubfolders)
	{ // will consider all subdirectories of *f
	  
	  for(cmonit_it added_path = addedContents.begin(); added_path != 
		addedContents.end(); ++added_path)
	    {
	      if(isSubdirectory(*f, added_path->first))
		checkAddedFolder(added_path, Dir, put_here);
	    }
	  
	}
      else
	{ // will only consider directory *f
	  cmonit_it added_path = addedContents.find(*f);
	  if(added_path != addedContents.end())
	    checkAddedFolder(added_path, Dir, put_here);
	}
      
    } // loop over folders to be watched
}

// check if added contents are tagged
void DaqMonitorBEInterface::checkAddedTags
(const rootDir & Dir, vector<MonitorElement*>& put_here) const
{
  for(cmonit_it added_path = addedContents.begin(); 
      added_path != addedContents.end(); ++added_path)
    checkAddedFolder(added_path, Dir, put_here);
}

// remove all CMEs
void DaqMonitorBEInterface::removeCollates()
{
  // empty collate_map first, to avoid 
  // calling CME dtor a 2nd time in DaqMonitorROOTBackEnd::removeElement
  collate_map.clear(); 

  for(cmesIt it = collate_set.begin(); it != collate_set.end(); ++it)
    delete (*it); // this also removes ME

  collate_set.clear();
}

// remove CME
void DaqMonitorBEInterface::removeCollate(CollateMonitorElement * cme)
{
  assert(cme);
  collate_map.erase(cme->getMonitorElement());
  collate_set.erase(cme);
  delete cme;
}

// remove all contents from <pathname> from all subscribers, tags and CMEs
void DaqMonitorBEInterface::removeCopies(const string & pathname)
{
  // we will loop over Subscribers, Tags and CMEs
  // and remove contents from all directories <pathname>
  for(sdir_it subs = Subscribers.begin(); subs!= Subscribers.end(); ++subs)
    { // loop over all subscribers
       MonitorElementRootFolder * dir = getDirectory(pathname, subs->second);
      // skip subscriber if no such pathname
      if(!dir)continue;
       removeContents(dir);
    } // loop over all subscribers
    

  for(tdir_it tag = Tags.begin(); tag != Tags.end(); ++tag)
    { // loop over all tags
      MonitorElementRootFolder * dir = getDirectory(pathname, tag->second);
      // skip tag if no such pathname
      if(!dir)continue;
      removeContents(dir);
   } // loop over all tags

  for(cmesIt cme = collate_set.begin(); cme != collate_set.end(); ++cme)
    { // loop over all CMEs
      MonitorElementRootFolder* dir=getDirectory(pathname,(*cme)->contents_);
      // skip CME is no such pathname
      if(!dir)continue;
      removeContents(dir);
    } // loop over all CMEs

}

// remove Monitor Element <name> from all subscribers, tags and CME directories
void DaqMonitorBEInterface::removeCopies(const string & pathname, 
					 const string & name)
{
  // we will loop over Subscribers, Tags and CMEs
  // and remove <name> from all directories <pathname>
  
  for(sdir_it subs= Subscribers.begin(); subs != Subscribers.end(); ++subs)
    // loop over all subscribers
    remove(pathname, name, subs->second);

  for(tdir_it tag = Tags.begin(); tag != Tags.end(); ++tag)
    // loop over all tags
    remove(pathname, name, tag->second);

  for(cmesIt cme = collate_set.begin(); cme != collate_set.end(); ++cme)
    // loop over all CMEs
    remove(pathname, name, (*cme)->contents_);
}

// remove Monitor Element <name> from <pathname> in <Dir>
void DaqMonitorBEInterface::remove(const string & pathname, 
				   const string & name, rootDir & Dir)
{
  MonitorElementRootFolder * dir = getDirectory(pathname, Dir);
  // skip subscriber if no such pathname
  if(!dir)return;
  
  removeElement(dir, name, false); // no warnings
}

// attach quality test <qc> to all ME matching <search_string>;
// if tag != 0, this applies to tagged contents
// <search_string> could : (a) be exact pathname (e.g. A/B/C/histo): FAST
// (b) include wildcards (e.g. A/?/C/histo, A/B/*/histo or A/B/*): SLOW
void DaqMonitorBEInterface::useQTest
(unsigned int tag, string search_string, const rootDir & Dir, QCriterion * qc) 
  const
{
  assert(qc);
  qc->add2search_path(search_string, tag);
  vME allMEs;
  scanContents(search_string, Dir, allMEs);
  addQReport(allMEs, qc); 
}

// attach quality test <qc> to directory contents ==> FAST
// if tag != 0, this applies to tagged contents
// (need exact pathname without wildcards, e.g. A/B/C);
// use flag to specify whether subfolders (and their contents) should be included;
void DaqMonitorBEInterface::useQTest(unsigned int tag, string pathname, 
				     bool useSubfolds, const rootDir & Dir, 
				     QCriterion * qc) const
{
  assert(qc);
  qc->add2folders(pathname, useSubfolds, tag);
  vME allMEs;
  if(useSubfolds)
    getAllContents(pathname, Dir, allMEs);
  else
    getContents(pathname, Dir, allMEs);
  addQReport(allMEs, qc);
}

// attach quality test <qtname> to tagged MEs ==> FAST
void DaqMonitorBEInterface::useQTest(unsigned int tag, const rootDir & Dir,
				     QCriterion * qc) const
{
  assert(qc);
  qc->add2tags(tag);
  vME allMEs;
  get(Dir.paths, allMEs);
  addQReport(allMEs, qc);
}


void DaqMonitorBEInterface::useQTest(string search_string, string qtname) const
{
   useQTest(0, search_string, qtname); // "0" means no tag
}   

void DaqMonitorBEInterface::useQTest(unsigned int tag, string search_string,
				    string qtname) const
{
  if(search_string.empty())
    return;

  QCriterion * qc = getQCriterion(qtname);
  if(!qc)
    {
      cerr << " *** Quality test " << qtname << " does not exist! " << endl;
      return;
    }

  if(tag == 0) // "0" means no tag
    useQTest(0, search_string, this->Own, qc);
  else
    {
      ctdir_it tg = Tags.find(tag);
      if(tg != Tags.end())
	useQTest(tag, search_string, tg->second, qc);
      else
	qc->add2search_path(search_string, tag);
    }
   
}

// attach quality test <qtname> to tagged MEs ==> FAST
// this action applies to all MEs already available or future ones
void DaqMonitorBEInterface::useQTest(unsigned int tag, string qtname) const
{
  QCriterion * qc = getQCriterion(qtname);
  if(!qc)
    {
      cerr << " *** Quality test " << qtname << " does not exist! " << endl;
      return;
    }
  if(tag == 0)
    {
      cerr << " *** Tag must be positive number! \n";
      return;
    }
  
  ctdir_it tg = Tags.find(tag);
  if(tg != Tags.end())
    useQTest(tag, tg->second, qc);
  else
    qc->add2tags(tag); 
}

// add quality reports to all MEs
void DaqMonitorBEInterface::addQReport(vector<MonitorElement *> & allMEs, 
				       QCriterion * qc) const
{
  assert(qc);
  string qr_name = qc->getName();
  for(vMEIt me = allMEs.begin(); me != allMEs.end(); ++me)
    {
      /* I need to double-check that qreport is not already added to ME;
	 This is because there is a chance that users may
	 1. define a ME after resetMonitoringDiff has been called
	 2. call MonitorUserInterface::useQTest
	 3. and then call MonitorUserInterface::runQTests, which
	 eventually calls this function
	 In this case ME appears in addedContents and this call would
	 give an error... (not sure of a better way right now)
      */
      if(!(*me)->getQReport(qr_name))
	    addQReport(*me, qc);

    }
}

// get "global" status (one of: STATUS_OK, WARNING, ERROR, OTHER) for group of MEs;
// returns most sever error, where ERROR > WARNING > OTHER > STATUS_OK;
// see Core/interface/QTestStatus.h for details on "OTHER" 
int DaqMonitorBEInterface::getStatus(const vector<MonitorElement *> & ME_group) 
  const
{
  if(hasError(ME_group))
    return dqm::qstatus::ERROR;
  else if(hasWarning(ME_group))
    return dqm::qstatus::WARNING;
  else if(hasOtherReport(ME_group))
    return dqm::qstatus::OTHER;
  else
    return dqm::qstatus::STATUS_OK;  
}

// true if at least one ME gave hasError = true
bool DaqMonitorBEInterface::hasError(const vector<MonitorElement *> & ME_group) 
  const
{
  for(vMEcIt me = ME_group.begin(); me != ME_group.end(); ++me)
    if(*me && (*me)->hasError())return true;
  // if here, no ME with an error has been found
  return false;

}

// true if at least one ME gave hasWarning = true
bool DaqMonitorBEInterface::hasWarning(const vector<MonitorElement *> & ME_group)
  const
{
  for(vMEcIt me = ME_group.begin(); me != ME_group.end(); ++me)
    if(*me && (*me)->hasWarning())return true;
  // if here, no ME with a warning has been found
  return false;
}

// true if at least one ME gave hasOtherReport = true
bool DaqMonitorBEInterface::hasOtherReport(const vector<MonitorElement *> & 
					   ME_group) const
{
  for(vMEcIt me = ME_group.begin(); me != ME_group.end(); ++me)
    if(*me && (*me)->hasOtherReport())return true;
  // if here, no ME with another (non-ok) status has been found
  return false;
}


const string DaqMonitorBEInterface::monitorDirName = "DQMData";
const string DaqMonitorBEInterface::referenceDirName = "Reference";
