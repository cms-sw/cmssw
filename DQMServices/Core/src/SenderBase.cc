#include "DQMServices/Core/interface/MonitorElementRootT.h"
#include "DQMServices/Core/src/DQMTagHelper.h"
#include "DQMServices/Core/src/SenderBase.h"
#include "DQMServices/Core/src/DQMMessage.h"
#include "DQMServices/Core/src/SocketUtils.h"

#include "Utilities/General/interface/MutexUtils.h"

#include <netinet/in.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <TObjString.h>

#include <iostream>

using namespace dqm::me_util;
using namespace dqm::monitor_data;
using namespace dqm::qtests;

using std::cout; using std::endl; using std::cerr;
using std::string; using std::vector;

// del: time delay in between shipments (in microsecs)
SenderBase::SenderBase(string name, unsigned del, bool pureProducer) : 
  NodeBase(name), pureProducer_(pureProducer), 
  del_(del), send_socket(-1)  
{
  mess_tmp = new DQMMessage;
  mess_tmp->setWhat (kMESS_OBJECT);
  updates = 0; buffer_ = 0;
  receiver_ = 0; 
}

SenderBase::~SenderBase(void)
{
  delete mess_tmp; 
}

// send list of monitorables; return success flag
bool SenderBase::sendMonitorable(void)
{
  bool success = true;
  // get monitorables here 
  vector<string> addedMonitorable;
  vector<string> removedMonitorable;
  // get tags here
  vector<string> addedTags;
  vector<string> removedTags;

  if(receiver_->newNode)
    {
      // if this is a source (pure-producer) send contents; else send monitorable
      if(pureProducer_)
	bei->getContents(addedMonitorable);
      else
	bei->getMonitorable(addedMonitorable);
 
      bei->getAllTags(addedTags);
      receiver_->newNode = false;
    }
  else
    {
      // allow for new monitorable elements to become available 
      // after initial connection
      if(pureProducer_)
	{
	  bei->getAddedContents(addedMonitorable);
	  bei->getRemovedContents(removedMonitorable);
	}
      else
	{
	  bei->getAddedMonitorable(addedMonitorable);
	  bei->getRemovedMonitorable(removedMonitorable);
	}

      bei->getAddedTags(addedTags);
      bei->getRemovedTags(removedTags);
    }
  
  if(addedMonitorable.empty() && removedMonitorable.empty()
     && addedTags.empty() && removedTags.empty())
    {
      cout << " *** Warning! No monitoring objects available! " << endl;
      return true;
    }

  string monlist; 
  try
    {
      // send removed monitorable to client, one directory at the time
      for(vIt it = removedMonitorable.begin(); 
	  it != removedMonitorable.end(); ++it)
	{
	  monlist = listRemPrefix + *it;
	  if(!sendMonString(monlist))success = false;
	}

      // send added monitorable to client, one directory at the time
      for(vIt it = addedMonitorable.begin(); it != addedMonitorable.end();
	  ++it)
	{
	  monlist = listAddPrefix + *it;
	  if(!sendMonString(monlist))success = false;
	}
      // send removed tags to client, one directory at the time
      for(vIt it = removedTags.begin(); 
	  it != removedTags.end(); ++it)
	{
	  monlist = tagRemPrefix + *it;
	  if(!sendMonString(monlist))success = false;
	}
      // send added tags to client, one directory at the time
      for(vIt it = addedTags.begin(); 
	  it != addedTags.end(); ++it)
	{
	  monlist = tagAddPrefix + *it;
	  if(!sendMonString(monlist))success = false;
	}

      // done sending monitorable, send "done" message to client
      DQMMessage mess1;
      mess1.setWhat (kMESS_DONE_MONITORABLE);
      int nerr = SocketUtils::sendMessage (&mess1, send_socket);
      if(nerr <= 0)
	{
	  cerr << " *** Error = " << nerr
	       << " while sending kMESS_DONE_MONITORABLE to " 
	       << receiver_->name << endl;
	  success = false;
	}
    } // try-block
  catch(...)
    {
      cout << " *** Failed to send monitorable to "<< receiver_->name << endl;
      success = false;
      throw;
    }

  return success;
}

// send monitorable string, return success flag
bool SenderBase::sendMonString(const string & sendThis)
{
  if(bei->DQM_VERBOSE)
    NodeBase::showMessage(" Sending monitorable to ", 
			  receiver_->name, sendThis);

  int nerr = SocketUtils::sendString (sendThis.c_str(), send_socket);
  if (nerr <= 0) 
    // Error message already printed by SocketUtils::checkedSocketWrite
    return false;
  return true;
}


// main "sending" loop: come here to send monitorable or monitoring
// return # of monitoring objects sent
int SenderBase::send()
{
  bool all_ok = shipMonitorable();
  if(!all_ok)return -1;
  
  return shipMonitoring();
}

// come here to send monitorable; return success flag
bool SenderBase::shipMonitorable()
{
  if(!send_socket) throw send_socket;

  if(shouldSendMonitorable())
    {
      if(!sendMonitorable())
	{
	  sendingFailed();
	  return false;
	}
    }

  return true;
}

// come here to send monitoring; return # of objects sent
int SenderBase::shipMonitoring()
{
  if(!send_socket) throw send_socket;

  // check requests pending from last cycle
  checkPendingRequests();
  int N = sendMonitoring(); // # of monitoring objects sent
  if (N < 0)
    sendingFailed();

  return N;
}

// send monitoring; return # of monitoring objects sent (<0 for errors)
int SenderBase::sendMonitoring(void)
{
  bool found_problems = false;
  int N_all = 0; // total # of monitoring objects sent

  try{

    // send monitoring, one directory/pathname at the time
    for(cdir_it path = receiver_->Dir->paths.begin(); 
	path != receiver_->Dir->paths.end(); ++path)
      { // loop over all pathnames of subscriber's folder structure

	//DO I NEED TO CLEAR mess_tmp??

	if(!path->second)
	  throw path->second;
	
	// for each pathname, pack monitoring elements into message
	int N = produceMonitoring(path->second);
	if(N < 0)
	  {
	    found_problems = true;
	    break;
	  }

	int nerr = 0;
	if(N)
	  {
	    std::ostringstream num; num << N;
	    // first send <pathname>:<# of objects in that directory> 
	    string mess_st = objectsPrefix + path->first + ":" + num.str();

	    // first send pathname and # of monitoring elements
	    nerr = SocketUtils::sendString (mess_st.c_str (), send_socket);
	    if(nerr <= 0)
	      {
		found_problems = true;
		break;
	      }

	    // now send the objects
	    nerr = SocketUtils::sendMessage (mess_tmp, send_socket);
	    if(nerr <= 0)
	      {
		found_problems = true;
		break;
	      }
	    else
	      N_all += N;
	  }

      } // loop over all pathnames of subscriber's folder structure

    if(N_all > 0)
      {
	// done sending monitoring; send "done" message
	DQMMessage mess2;
	mess2.setWhat (kMESS_DONE_MONIT_CYCLE);
	
	int nerr = SocketUtils::sendMessage (&mess2, send_socket);
	if (nerr <= 0)
	  {
	    found_problems = true;
	  }

      }
    
  } // try-block
  catch (MonitorElement * me)
    {
      nullError("MonitorElement");
      found_problems = true;
    }
  catch (ME_map * me)
    {
      nullError("ME_map");
      found_problems = true;
    }
  catch(...)
    {
      cerr << " *** Error! Unknown exception " << endl; 
      found_problems = true;
      throw;
    }

  if(found_problems)
    {
      cout << " Error caught while sending monitoring to client " 
	   << receiver_->name << endl;
      return -1;
    }

  if(N_all)
    {
      ++updates;
      ++receiver_->count;
      receiver_->n_failed_consec = 0;
      
      if(bei->DQM_VERBOSE)
	{
	  if(updates % NodeBase::printout_period(updates) == 0) 
	    {
	      sockaddr_in addr;
	      socklen_t addrlen = sizeof (sockaddr_in);
	      getpeername (send_socket, (sockaddr *) &addr, &addrlen);
	      hostent *hp = gethostbyaddr ((char *) &addr.sin_addr, 
					   sizeof (addr.sin_addr), 
					   addr.sin_family);
	      char* hostname;
	      if (hp == NULL)
		hostname = inet_ntoa (addr.sin_addr);
	      else
		hostname = hp->h_name;
	      
	      cout << " # of monitoring packages sent: " << updates
		   <<"\n (last one was " << N_all 
		   << " monitoring elements sent to " 
		   << hostname << ") " 
		   << endl;
	    }
	}
      
    }
  
  return N_all;
}

// receive subscription from receiver
void SenderBase::getSubscription(void)
{
  if(!send_mess->what ()) return;

  // if here, node has sent non-zero message
  if(send_mess->what() == kMESS_STRING)
    {
      // string message has been read in by inheriting class and stored in buffer_

      // ------------------------------------------
      // Node has sent subscription request
      // ------------------------------------------
      if(buffer_->find(subscAddPrefix) != string::npos)
	addSubscription(buffer_->c_str() + subscAddPrefix.size());
      // ------------------------------------------
      // Node has sent un-subscription request
      // ------------------------------------------
      else if(buffer_->find(subscRemPrefix) != string::npos)
	removeSubscription(buffer_->c_str() + subscRemPrefix.size());
      // ------------------------------------------
      // Node has sent an uknown message
      // ------------------------------------------
      else if(buffer_->find(nameClientPrefix) != string::npos)
	{
	  // ------------------------------------------
	  // Collector has sent its name (only for sources)
	  // ------------------------------------------
	  TString name = buffer_->c_str() + nameClientPrefix.size();
	  string sname = name.Data();
	  cout << " Connected node identified itself as " << sname << endl;
	  receiver_->name = sname;
	}
      else
 	{
	  cout << " Node " << receiver_->name << " says: " << *buffer_
	       << endl;
	  cout << " Message is not understood. Ignoring... " << endl;
	}
      
    } // message is a string

  else if(send_mess->what() == kMESS_DONE_SUBSCRIPTION)
    {
      receiver_->need2readSubscription = false; // done reading subscription
      prepareRecvSubscription();
   }
  else
    {
      cerr << " *** Error! Client " << receiver_->name 
	   << " has sent unknown message of type " << send_mess->what() 
	   << endl;
    }
  buffer_ = 0;
}

// add objects to receiver's subscription list; 
// "name" format: see DirFormat definition
// if new_request = true, send request to higher-level class if needed
// return success flag
bool SenderBase::addSubscription(const string & name, bool new_request)
{
  if(new_request && bei->DQM_VERBOSE)
    NodeBase::showMessage(" Processing subscription request from ",
			  receiver_->name, name);

  // unpack "name" into dir_path and contents
  DirFormat subsc;
  if(!unpackDirFormat(name, subsc))
    return false;

  // keep this flag true until Receiver sends "kMESS_DONE_SUBCRIPTION"
  if(new_request)
    receiver_->need2readSubscription = true;

  // get monitorable folder;
  MonitorElementRootFolder * dir_orig = 0;
  if(subsc.tag)
    { // subscription request contains a tag
      tdir_it tg;
      if(bei->tagHelper->getTag(subsc.tag, tg, false)) // do not create
	dir_orig = bei->getDirectory(subsc.dir_path, tg->second);
    }
  else // subscription request does NOT contain a tag
    dir_orig = bei->getDirectory(subsc.dir_path);

  // folder w/ monitoring information is missing (maybe some error?)
  if(!dir_orig)
    {
      // save subscription request
      if(new_request)saveRequest2add(name);
      return false;
    }
  // if the name is of the form "<dir>:" this implies all the contents
  bool allContents = subsc.contents.empty();
  if(allContents)
    {
      if(!getFullContents(dir_orig, subsc))
	return false;
    }

  string missing_items;
  bool copiedAny = copy(dir_orig, subsc, missing_items);
  if(new_request)
    {
      // if no elements were found and all contents were implied
      if(!copiedAny && allContents)
	{
	  // save original subscription request
	  saveRequest2add(name);
	  return false;
	}
      // if there are elements that were not found
      if(!missing_items.empty())
	// save modified subscription request
	saveRequest2add(missing_items);
    }
  
  // success only if all monitoring elements were found
  return missing_items.empty();
}

// remove objects from receiver's subscription list
// "name" format: see DirFormat definition
// return success flag
bool SenderBase::removeSubscription(const string & name)
{
  if(bei->DQM_VERBOSE)
    NodeBase::showMessage(" Processing subscription cancellation request from ",
			  receiver_->name, name);

  // unpack "name" into dir_path and contents
  DirFormat dir;
  if(!unpackDirFormat(name, dir))
    return false;

  MonitorElementRootFolder * dir_subsc = 
    bei->getDirectory(dir.dir_path, *(receiver_->Dir));
  if(!dir_subsc)
    {
      cerr << " *** Error! Folder " << dir.dir_path << " does not exist!"
	   << endl;
      cerr << " Ignored request to unsubscribe " << endl;
      return false;
    }

  // if unsubscription request contains empty path with tag
  // update dir.contents with tagged MEs
  if(dir.tag && dir.contents.empty())
    {
      vME all_tagged_MEs;
      bei->getContents(dir.dir_path, dir.tag);
      // loop over all tagged MEs
      for(vME::const_iterator it = all_tagged_MEs.begin(); 
	  it != all_tagged_MEs.end(); ++it)
	{
	  // check if ME belongs to subscriber
	  if(dir_subsc->findObject((*it)->getName()) )
	    dir.contents.push_back((*it)->getName());
	}
    }

  bei->removeSubsc(dir_subsc, dir.contents);  

  checkIfNeeded(dir);
  // keep this flag true until Receiver sends "kMESS_DONE_SUBCRIPTION"
  receiver_->need2readSubscription = true;
  return true;
}

// put updated monitoring objects into (private member) mess
// return # of objects to be sent out
int SenderBase::produceMonitoring(MonitorElementRootFolder * folder)
{
  string pathname = folder->getPathname();

  vME objects;
  folder->getContents(objects);

  int foundUpdates = 0;
  DQMRootBuffer *buffer = new DQMRootBuffer (TBuffer::kWrite);
  for(vME::iterator it = objects.begin(); it != objects.end(); ++it)
    { // loop over monitoring objects in subscriber's current folder

      if(!(*it))
	throw (*it);

      // send out monitoring element if (a) was updated (b) it has never been
      // sent to this particular subscriber (ie. new request)
      bool wasUpdated = (*it)->wasUpdated();
      bool neverSent = isNeverSent(folder, (*it)->getName());
      if(wasUpdated || neverSent)
	{ 
	  LockMutex a(((*it))->mutex); // lock it
	  
	  try
	    {
	      MonitorElementRootObject* ob = 
		dynamic_cast<MonitorElementRootObject *>((*it));
	      if(ob)
		{
		  buffer->WriteObject(ob->operator->()); 
		}
	      else
		{
		  sendTObjString((*it), buffer);
		}

	    } // try-block
	  catch (...)
	    {
	      cerr << " *** Error extracting monitoring object!" << endl;
	      continue;
	    }

	  if(wasUpdated)
	    bei->add2UpdatedContents((*it)->getName(), pathname);

	  // should we set flag to false after TMessage has been sent out?
	  if(neverSent)
	    setNeverSent(folder, (*it)->getName(), false);

	  ++foundUpdates;

	} // will send out monitoring element

      foundUpdates += sendQReports((*it), folder, buffer);

    } // loop over monitoring elements in directory
  // if(buffer->Length () != 0)
  mess_tmp->setBuffer (buffer, buffer->Length ());

  return foundUpdates;
}

// make directory structure (for receiver; to be called by inheriting class)
rootDir * SenderBase::makeDirStructure(const string & name)
{

  sdir_it it = bei->Subscribers.find(name);
  if(it == bei->Subscribers.end())
    {
      // name not appearing in Subscribers; must construct root folder
      std::pair<sdir_it, bool> newEntry;
      newEntry = bei->Subscribers.insert 
	(subscriber_map::value_type(string(name), rootDir()) );

      if(newEntry.second) // <bool flag> = true for success
	it = newEntry.first;
      else
	{
	  cerr << " *** Failed to construct root directory for subscriber "
	       << name << endl;
	  return (rootDir *) 0;
	}
    }

  if(!it->second.top)
    // root folder for subscriber does not exist: create now
    bei->makeDirStructure(it->second, name);

  return &(it->second);
}

// come here for cleanup when a receiver goes down;
// cleanup method to be called by inheriting class when receiver is disconnected
void SenderBase::cleanupReceiver(void)
{

  // store here directories that correspond to removed (subscriber's) directories 
  vector<string> check_dirs;

  // send monitoring, one directory/pathname at the time
  for(dir_it path = receiver_->Dir->paths.begin(); 
      path != receiver_->Dir->paths.end(); ++path)
    { // loop over all pathnames of subscriber's folder structure
      
      // skip root directory (only interested in directories
      // that correspond to (upstream) senders/sources
      if(isTopFolder(path->first))
	continue;

      // store string of the form: <dir_name>:
      check_dirs.push_back(path->first + ":");
    }

  // remove directory corresponding to subscriber
  bei->rmdir(ROOT_PATHNAME, *(receiver_->Dir));

  // remove requests to add (since receiver has gone down)
  receiver_->request2add.clear(); // is this really necessary?
  // remove entry from subscribers structure
  bei->Subscribers.erase(receiver_->name);


  DirFormat dir;
  // check directories to determine if contents are needed after receiver went down
  for(cvIt it = check_dirs.begin(); it != check_dirs.end(); ++it)
    {
      if(unpackDirFormat(*it, dir))
	checkIfNeeded(dir);
    }

  // push requests to remove (if any)
  prepareRecvSubscription();

}

// come here for cleanup when sender goes down;
// cleanup method to be called by inheriting class when sender is disconnected
void SenderBase::cleanupSender(const string & name)
{
  // we want to remove <name> folder in subscriber's structure
  if(bei->getDirectory(name, *(receiver_->Dir)))
    bei->rmdir(name, *(receiver_->Dir));

  // remember to remove references to histograms created for sender <name>
  // (see e.g. ReceiverBase::cleanupSender(void) )
  string hsizename = name + "_size";
  string hfreqname = name + "_updel";

  bool warning = false;
  bei->removeElement(receiver_->Dir->top, hsizename, warning);
  bei->removeElement(receiver_->Dir->top, hfreqname, warning);
}

// check if requests from downstream class have been fullfilled
void SenderBase::checkPendingRequests(void)
{
  lIt it; vector<lIt> matches;
  for(it = receiver_->pend_request.begin(); 
      it != receiver_->pend_request.end(); ++it)
    {
      if(!addSubscription(*it, false)) // do not create new request
	{
	  // show warning if request still pending...
	  if(bei->DQM_VERBOSE)
	    NodeBase::showMessage(" Pending request to add subscription",
				  "", *it);
	}
      else
	matches.push_back(it);
    }

  for(vector<lIt>::iterator it2 = matches.begin(); it2 != matches.end(); 
      ++it2)
    receiver_->pend_request.erase(*it2);
}

// come here to save subscription request for upstream class
void SenderBase::saveRequest2add(const string & name)
{
  if(!pureProducer_) // request to add (only if this is not a source!)
    // save request in original format
    receiver_->request2add.push_back(name);
  else
    {
      cerr << " *** Error! Incomprehensible subscription request: <" 
	   << name << "> from node " << receiver_->name << endl;
      cerr << " Ignored request to subscribe " << endl;
    }
 
}

// copy monitoring elements (described in subscription request: <subsc>)
// from directory <dir_orig>; create new request <missing_items> with objects
// that were not found in <dir_orig>;
// return true if at least one ME was copied
bool SenderBase::copy(const MonitorElementRootFolder * dir_orig, const DirFormat
		      & subsc, string & missing_items)
{
  vector<string> available; available.clear();
  // loop over objects of subscription request
  for(cvIt it = subsc.contents.begin(); it != subsc.contents.end(); ++it)
    {
      if(!dir_orig->hasMonitorable(*it))
	{
	  cerr << " *** Folder " << dir_orig->getPathname() 
	       << " has no ME named " << *it << endl;
	  cerr << " Subcription request is ignored... " << endl;
	  continue;
	}

      if(dir_orig->findObject(*it))
	// object is available
	available.push_back(*it);
      else
	// object not available: add to requests
	{
	  if(!missing_items.empty())missing_items += ",";
	  missing_items += *it;
	} // object not available: add to requests

    }// loop over objects of subscription request

  if(!available.empty())
    {
      MonitorElementRootFolder * dir_subsc = 
	bei->makeDirectory(subsc.dir_path, *(receiver_->Dir));
      bei->copy(dir_orig, dir_subsc, available);
    }
    
  // things like <dir>: (implying all contents) 
  // have been taken care of in addSubscription
  if(!missing_items.empty())
    missing_items = subsc.dir_path + ":" + missing_items;
  
  return(!available.empty());
}

// true if we should send monitorable to receiver
bool SenderBase::shouldSendMonitorable(void) const
{
  // we should send monitorable if (a) this is a new node
  // (b) monitorable has been updated
  // or (c) tags have been updated
  if(receiver_->newNode)return true;

  bool newMonitorable = false;
  if(pureProducer_)
    newMonitorable = !bei->addedContents.empty() || 
      !bei->removedContents.empty();
  else
    newMonitorable = !bei->addedMonitorable.empty() || 
      !bei->removedMonitorable.empty();

  if(newMonitorable)return true;

  bool modifiedTags = !bei->addedTags.empty() || 
    !bei->addedTags.empty();
  
  return modifiedTags; 
}

#include <unistd.h>

/// doneSendingMonitorable & doneSendingMonitoring combined
void SenderBase::doneSending(bool resetMEs, bool callResetDiff)
{
  bei->doneSendingMonitoring(resetMEs, callResetDiff);
  bei->doneSendingMonitorable(callResetDiff);
  bei->unlock();
  // done sending; now, take a break
  usleep(del_);
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
void SenderBase::doneSendingMonitoring(bool resetMEs, bool callResetDiff)
{
  bei->doneSendingMonitoring(resetMEs, callResetDiff);
  bei->unlock();
}

/* come here after sending monitorable to all receivers;
   if callResetDiff = true, call resetMonitorableDiff
   (typical behaviour: Sources & Collector have callResetDiff = true, whereas
   clients have callResetDiff = false, so GUI/WebInterface can access the 
   modifications in monitorable & monitoring) */
void SenderBase::doneSendingMonitorable(bool callResetDiff)
{
  bei->doneSendingMonitorable(callResetDiff);
  bei->unlock();
  // done sending; now, take a break
  usleep(del_);
}
// come here at beginning of monitoring cycle
void SenderBase::startSending(void)
{
  /* We MUST lock the bei before we start shipping monitoring, because if there
     are more than two receivers and a ME gets updated after we are done sending
     to receiver #1, but before sending to receiver #2, receiver #1 will never
     receive a fresh copy of ME (method resetUpdate is called at doneSending)
   */
  bei->lock();
}

// come here when attempt to send monitorable/monitoring fails
void SenderBase::sendingFailed(void)
{
  ++receiver_->n_failed;
  ++receiver_->n_failed_consec;
}

// check if objects appearing in unsubscription request <dir> are needed
// by any other subscribers (if any); if not, will issue unsubscription request
// to upstream class (if any)
void SenderBase::checkIfNeeded(DirFormat & dir)
{
  // this is not relevant for sources, only for collectors...
  if(pureProducer_)return;

  MonitorElementRootFolder* folder = bei->getDirectory(dir.dir_path);
  // if we have an empty dir (implying all contents), create a list of them
  // so we can loop over them...
  if(dir.contents.empty())
    getFullContents(folder, dir);
  
  string unsub_request;
  // loop over monitoring elements
  for(cvIt it = dir.contents.begin(); it != dir.contents.end(); ++it)
    {
      // check if Monitoring Element is needed by any subscriber
      if(!bei->isNeeded(dir.dir_path, *it))
	{
	  bei->removeElement(folder, *it);
	  // add to unsubscription request
	  if(!unsub_request.empty())unsub_request += ",";
	  unsub_request += *it;
	}

    } // loop over monitoring elements

  // have come up with unsubscription request
  if(!unsub_request.empty())
    {
      unsub_request = dir.dir_path + ":" + unsub_request;
      receiver_->request2remove.push_back(unsub_request);
    }

}

// prepare subscription requests for higher-level class (if applicable)
void SenderBase::prepareRecvSubscription(void)
{
  if(pureProducer_)return;

  {
    LockMutex a(bei->requests.mutex);

    // loop over requests to add, remove
    for(lIt it = receiver_->request2add.begin(); 
	it != receiver_->request2add.end(); ++it)
      {
	bei->requests.toAdd.push_back(*it);
	// save pending requests here
	receiver_->pend_request.push_back(*it);
      }

    for(lIt it = receiver_->request2remove.begin(); 
	it != receiver_->request2remove.end(); ++it)
      bei->requests.toRemove.push_back(*it);
  }
  
  receiver_->request2add.clear();
  receiver_->request2remove.clear();
}
 
// true if ME in folder has never been sent to subscriber
bool SenderBase::isNeverSent(MonitorElementRootFolder * folder, string ME_name)
{
  if(folder->neverSent.find(ME_name) == folder->neverSent.end())
    { // first time flag is accessed for this ME
      folder->neverSent[ME_name] = neverSent_data(); // initialize to true
    }

  return folder->neverSent[ME_name].me;
}

// true if QReport for ME in folder has never been sent to subscriber
bool SenderBase::isNeverSent(MonitorElementRootFolder * folder, 
			     string ME_name, string qtname)
{
  // first, look for ME
  if(folder->neverSent.find(ME_name) == folder->neverSent.end())
    { // first time flag is accessed for this ME
      folder->neverSent[ME_name] = neverSent_data();
    }

  // now look for QReport
  if(folder->neverSent[ME_name].qr.find(qtname) == 
     folder->neverSent[ME_name].qr.end())
    {
      // first time flag is accessed for this QReport
      folder->neverSent[ME_name].qr[qtname] = true;
    }

  return folder->neverSent[ME_name].qr[qtname];
}

// set "neverSent" flag for ME in folder of subscriber
void SenderBase::setNeverSent(MonitorElementRootFolder * folder, 
			      string ME_name, bool flag)
{
  if(folder->neverSent.find(ME_name) == folder->neverSent.end())
    { // first time flag is accessed for this ME
      folder->neverSent[ME_name] = neverSent_data(); // initialize to true
    }

  folder->neverSent[ME_name].me = flag;
}

// set "neverSent" flag for QReport of ME in folder of subscriber
void SenderBase::setNeverSent(MonitorElementRootFolder * folder, 
			      string ME_name, string qtname, bool flag)
{
  // first, look for ME
  if(folder->neverSent.find(ME_name) == folder->neverSent.end())
    { // first time flag is accessed for this ME
      folder->neverSent[ME_name] = neverSent_data(); 
    }
  
  folder->neverSent[ME_name].qr[qtname] = flag;

}

// send QReports associated with <me>; return total # of reports sent
unsigned SenderBase::sendQReports(MonitorElement * me, 
				  MonitorElementRootFolder * folder,
				  DQMRootBuffer *buffer)
{
  unsigned ret = 0;
  if(!me)
    {
      cerr << " Null Monitoring Element in SenderBase::sendQReports! " << endl;
      return ret;
    }
  
  QR_map qreports = me->getQReports();
  for(qr_it it = qreports.begin(); it !=  qreports.end(); ++it)
    { // loop over ME's qreports
      MERootQReport * mqr = dynamic_cast<MERootQReport *> (it->second);
      if(!mqr)
	continue;

      bool wasUpdated = mqr->wasUpdated();
      bool neverSent = isNeverSent(folder,me->getName(), mqr->getQRName());
      
      if(wasUpdated || neverSent)
	{
	  LockMutex a(mqr->mutex); // lock it
	  
	  try
	    {
	      sendTObjStringQ(mqr, buffer);
	    }
	  catch (...)
	    {
	      cerr << " *** Error extracting quality report!" << endl;
	      continue;
	    }

	  if(wasUpdated)
	    bei->add2UpdatedQReports(mqr); 
	  // should we set flag to false after TMessage has been sent out?
	  if(neverSent)
	    setNeverSent(folder, me->getName(), mqr->getName(), false);
      
	  ++ret;
	  
	}

    } // loop over ME's qreports

  return ret;
}

// true if receiver is done sending subscription requests
bool SenderBase::isReceiverDone(void) const 
{
  // if no receiver, there aren't any subscription requests either!
  if (!receiver_)
    return true;
  
  return !receiver_->need2readSubscription;
}

// send TObjString from MonitorElement
void SenderBase::sendTObjString(MonitorElement * me, DQMRootBuffer *buffer)
{
  if(!me)return;

  FoldableMonitor * fm = dynamic_cast<FoldableMonitor *> (me);
  if(fm)
    buffer->WriteObject(fm->getTagObject());
  else
    cerr << " *** Failed to extract and send object " 
	 << me->getName() << endl;
}

// send TObjString from MERootQReport
void SenderBase::sendTObjStringQ(MERootQReport * me, DQMRootBuffer *buffer)
{
  TObjString * ts = me->getTagObject();
  if(ts)
    buffer->WriteObject(ts);
  else
    {
      cerr << " *** Failed to extract and send object " 
	   << me->getName() << endl;
    }
}
