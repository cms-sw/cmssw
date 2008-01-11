#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Core/interface/DQMTagHelper.h"
#include "DQMServices/Core/interface/ReceiverBase.h"
#include "DQMServices/Core/interface/SocketUtils.h"
#include "DQMServices/Core/interface/DQMMessage.h"
//#include "DQMServices/Diagnostic/interface/TimeMonitor.h"
//include if you want diagnostic stuff to be active
//#include "DQMServices/Core/interface/ProfileDiagnostic.h" 

#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <netdb.h>
#include <errno.h>
#include <arpa/inet.h>

#include <string>
#include <set>

using namespace dqm::me_util;
using namespace dqm::monitor_data;

using std::cout; using std::endl; using std::cerr;
using std::string; using std::vector; using std::set;

ReceiverBase::ReceiverBase(string name, bool keepStaleSources) : 
  NodeBase(name), timechart(0), meanupdel(0.), updelds(0.), tx(0.), 
  updates(0), recv_mess(0), recv_socket(-1),  sender_(0), 
  keepStaleSources_(keepStaleSources)
{
  N_me_recv = 0; buffer_ = 0;
//#ifdef DO_DIAGNOSTIC 
//  timeMonitor = new TimeMonitor ();
//#endif
  recv_mess = new DQMMessage;
  N_me_recv = 0; buffer_ = 0;

}

ReceiverBase::~ReceiverBase(void)
{
//#ifdef DO_DIAGNOSTIC
//  delete timeMonitor;
//#endif
}

// main subscription loop
void ReceiverBase::doSubscription(void)
{
  // subscribe, if possible
  if(canSubscribe())produceSubscription();
  // send subscription, if necessary
  if(shouldSendSubscription())
    sendSubscription();
}

// main "receiving" loop: 
// come here once message has been received (in higher-level class); 
// return kMESS_DONE_MONIT_CYCLE if done w/ monitoring cycle, 0 otherwise
int ReceiverBase::receive(void)
{
  if (recv_socket < 0) throw recv_socket;
  if (!sender_) throw sender_;
  if (!recv_mess->what ()) 
    return 0;       ///UNUSEFUL CHECK IT
  
  unsigned message_type = recv_mess->what ();

  if (message_type == kMESS_STRING)
    processString ();
  else if (message_type == kMESS_OBJECT)
    {
//#ifdef DO_DIAGNOSTIC
      //TIME MONITOR
//      timeMonitor->TimeMonitor::set (sender_->name.c_str(), 
//				     seal::TimeInfo::time (), 
//				     recv_mess->length ());
//#endif
      processObject ();
    }
  
  //  else if(message_type == kMESS_DONE_MONIT_DIR)
  // done receiving monitoring information for current directory
  //
  else if(message_type == kMESS_DONE_MONIT_CYCLE)
    {
//#ifdef DO_DIAGNOSTIC
//      timeMonitor->TimeMonitor::setDone (sender_->name.c_str ());	
      //TIME MONITOR
//      timeMonitor->TimeMonitor::set (sender_->name.c_str(), 
//				     seal::TimeInfo::time ()); 	
      //TIME MONITOR
//#endif
      sender_->need2readMonitoring = false; 
      ++updates;
      ++sender_->count;
      if(bei->DQM_VERBOSE)
	{
	  if (updates % NodeBase::printout_period(updates) == 0)
	    {
	      sockaddr_in addr;
	      socklen_t addrlen = sizeof (sockaddr_in);
	      getpeername (recv_socket, (sockaddr *) &addr, &addrlen);
	      hostent *hp = gethostbyaddr ((char *) &addr.sin_addr, 
					   sizeof (addr.sin_addr), 
					   addr.sin_family);
	      char* hostname;
	      if (hp == NULL)
		hostname = inet_ntoa (addr.sin_addr);
	      else
		hostname = hp->h_name;
	      
	      cout << " # of monitoring packages received: " << updates 
		   << "\n (last one was " << N_me_recv 
		   << " monitoring elemenents received from " 
		   << hostname << ") " 
		   << endl;
	    }
	}
      N_me_recv = 0;
      // done receiving monitoring information for all directories
       return kMESS_DONE_MONIT_CYCLE;
    }
  else if(message_type == kMESS_DONE_MONITORABLE)
    {
      sender_->need2readMonitorable = false; // done receiving monitorable
      return kMESS_DONE_MONITORABLE;
    }
  else 
    {
      cerr << " *** Error! Unexpected message type " << message_type 
	   << " from node " << sender_->name << endl;
      throw false;
    }

  return 0;
 }

// process string messages
void ReceiverBase::processString(void)
{
  // string message has been read in by inheriting class and stored in buffer_

  // ------------------------------------------
  // Sender has sent a list of monitorables
  // ------------------------------------------
  // to add
  if(buffer_->find(listAddPrefix) != string::npos)
    addMonitorable(buffer_->c_str() + listAddPrefix.size());
  // to remove
  else if(buffer_->find(listRemPrefix) != string::npos)
    removeMonitorable(buffer_->c_str() + listRemPrefix.size());
  // ------------------------------------------
  // Sender has sent description of list of monitoring objects
  // ------------------------------------------
  else if(buffer_->find(objectsPrefix) != string::npos)
    getObjectDesc(buffer_->c_str() + objectsPrefix.size());
  // ------------------------------------------
  // Sender has sent description of tags
  // ------------------------------------------
  // to add
  else if(buffer_->find(tagAddPrefix) != string::npos)
    modifyTags(buffer_->c_str()+tagAddPrefix.size(), true); // add tags
  // to remove
  else if(buffer_->find(tagRemPrefix) != string::npos)
    modifyTags(buffer_->c_str()+tagRemPrefix.size(), false); // remove tags
  // ------------------------------------------
  // Sender has sent an uknown message
  // ------------------------------------------
  else
    {
      cout << " Node " << sender_->name << " says: " << *buffer_ << endl;
      cout << " Message is not understood. Ignoring... " << endl;
    }

  buffer_ = 0;
}

// process object messages
void ReceiverBase::processObject(void)
{
  float msize  = float(recv_mess->buffer ()->BufferSize());

  // get hold of histogram pointers here
  getMessageSizeHist(msize);
  getUpdFreqHist();
  sender_->size->Fill(msize);

  // directory for current objects: 
  // <sender's name> + pathname of monitoring objects
  string subdir = sender_->name;
  if(!isTopFolder(sender_->path))
    subdir += "/" + sender_->path;

  bool stay_in_loop = true;

  while(sender_->objn > 0 && stay_in_loop)
    {
      sender_->objn--;
      
      string obj_name = recv_mess->getClass()->GetName();
      if(obj_name.find("TH") == string::npos && 
	 obj_name.find("TP") == string::npos && 
	 obj_name.find("TO") == string::npos)
	{
	  cerr << " *** Error! Unknown object " 
	       << recv_mess->getClass()->GetName() << endl;
	  continue;
	}
      stay_in_loop = extractObject(subdir);
      if(stay_in_loop)
	++N_me_recv;
    } // stay here till all objects have been read

} 

// extract object (TH1F, TH2F, ...) from message; return success flag
bool ReceiverBase::extractObject(string pathname)
{
  DQMRootBuffer * buf = recv_mess->buffer ();
  TObject * to = buf->ReadObject(recv_mess->getClass());
  if(!to)
    {
      cout << " *** Erorr! Failed to read in object from node " 
	   << sender_->name << endl;
      return false;
    }
  
  MonitorElementRootFolder * dir = bei->getDirectory(pathname);
  const bool fromRemoteNode = true; // TObject arriving from upstream node
  bool success = bei->extractObject(to, dir, fromRemoteNode);

  delete to;
  return success;
}

// get update time chart histogram
void ReceiverBase::getTimeChartHist(void)
{
  string upd = "updchart";
  MonitorElement * me = bei->findObject(upd, ROOT_PATHNAME);
  if(!me)
    {
      MonitorElementRootFolder * dir = bei->getDirectory(ROOT_PATHNAME);
      me = bei->book1D(upd, "update time chart", 5000, 0., 5000.,dir);
    }
  if(!timechart)
    {
      timechart = convertObject<TH1F, MonitorElementRootH1>(me); 
      timechart->SetMinimum(0.);
      timechart->SetMaximum(20.);
    }
}

// get message size histogram for socket
 void ReceiverBase::getMessageSizeHist(float msize)
{
  string hsizename = sender_->name + "_size";
  MonitorElement * me = bei->findObject(hsizename, ROOT_PATHNAME);
  if(!me)
    {
      MonitorElementRootFolder * dir = bei->getDirectory(ROOT_PATHNAME);

      string hsiztitle = sender_->name + " Message Size Distribution";
      // find a smarter way to change the range and rebin when obsolete.
      me = bei->book1D(hsizename, hsiztitle,100, msize/4., 2.*msize,dir); 
    }
  sender_->size = convertObject<TH1F, MonitorElementRootH1>(me); 
}

// get update frequency histogram for socket
void ReceiverBase::getUpdFreqHist()
{
  string hfreqname = sender_->name + "_updel";
  MonitorElement * me = bei->findObject(hfreqname, ROOT_PATHNAME);
  if(!me)
    {
      MonitorElementRootFolder * dir = bei->getDirectory(ROOT_PATHNAME);

      string hfreqtitle = " Update frequency per node";
      me = bei->book1D(hfreqname, hfreqtitle, 100, 0., 20., dir);
      sender_->timer = TTimeStamp();
    }
  sender_->freq = convertObject<TH1F, MonitorElementRootH1>(me); 
}

// remove monitoring elements associated with sender_; 
// to be called by inheriting class when sender is disconnected
void ReceiverBase::cleanupSender(void)
{
  // remove monitoring only if we do not keep stale sources
  cleanup(sender_->name, !keepStaleSources_);
}

// remove monitoring elements associated with (obsolete) sender_ (with same name);
// to be called by inheriting class when sender is (re)connected
void ReceiverBase::cleanupObsoleteSender(void)
{
  // remove monitoring only if there may be an obsolete directory 
  // (with same name) from previous connection
  cleanup(sender_->name, keepStaleSources_);
}
    
// to be called by cleanupSender and cleanupObsoleteSender only
void ReceiverBase::cleanup(string name, bool doIt)
{
  if(doIt)
    {
      // Delete the whole "sender" directory (w/ objects sent from sender_)
      removeDir(name);
      // show updated dir structure
      if(bei->getVerbose())bei->showDirStructure();
    }
}

// remove directory, update monitorable
void ReceiverBase::removeDir(string subdir)
{
  if(bei->getDirectory(subdir))
    {
      bei->rmdir(subdir);
      lessMonitorable(subdir); // remove <subdir> from monitorable  
    }
}

// return <sender's name>_is_dead
string ReceiverBase::getDeadName(void) const
{
  string name = "Unknown";
  if(sender_)name = sender_->name;
  return name + "_is_dead";
}

// return <sender's name>_is_done
string ReceiverBase::getDoneName(void) const
{
  string name = "Unknown";
  if(sender_)name = sender_->name;
  return name + "_is_done";
}

// create string at top directory indicating sender is done
void ReceiverBase::senderIsDone(void)
{

  // create a string that shows "finished" sender...
  string name = getDoneName();
  
  MonitorElementRootFolder * dir = (MonitorElementRootFolder *)
  bei->getDirectory(ROOT_PATHNAME);

  bei->bookString(name, "Automated message", dir);
  moreMonitorable(ROOT_PATHNAME); // add <name> to monitorable
}

// create string at top directory indicating sender is dead
void ReceiverBase::senderIsDead(void)
{
  // create a string that shows problematic sender...
  string name = getDeadName();

  if(sender_ && sender_->name != DummyNodeName)
    {
      MonitorElementRootFolder * dir = bei->getDirectory(ROOT_PATHNAME);
      bei->bookString(name, "Automated message", dir);
      moreMonitorable(ROOT_PATHNAME); // add <name> to monitorable
    }
}
 
// reverse action of senderIsDead
void ReceiverBase::senderIsNotDead(void)
{
  // see if there is a string that shows a problematic sender...
  string name = getDeadName();
  if(bei->findObject(name, ROOT_PATHNAME))
    {
      MonitorElementRootFolder * dir = bei->getDirectory(ROOT_PATHNAME);
      bei->removeElement(dir, name);
      lessMonitorable(ROOT_PATHNAME); // remove <name> from monitorable
    }
}

// reverse action of senderIsDone
void ReceiverBase::senderIsNotDone(void)
{
  // see if there is a string that shows a "finished" sender...
  string name = getDoneName();
  if(bei->findObject(name, ROOT_PATHNAME))
    {
      MonitorElementRootFolder * dir = bei->getDirectory(ROOT_PATHNAME);
      bei->removeElement(dir, name);
      lessMonitorable(ROOT_PATHNAME); // remove <name> from monitorable
    }
}


// add objects to monitorable; "name" format: see DirFormat definition
// return success flag
bool ReceiverBase::addMonitorable(const string & name)
{
  if(bei->DQM_VERBOSE)
    NodeBase::showMessage(" Monitorable addition from node ",
			  sender_->name, name);

  // unpack "name" into dir_path and contents
  DirFormat dir;
  if(!unpackDirFormat(name, dir))
    return false;

  string dir_path = sender_->name;
  // new_name will hold the (modified: sender_->name + <old name>) monitorable
  string new_name;
  // add subdirectory (unless it is the top folder)
  if(!isTopFolder(dir.dir_path))
    {
      dir_path += "/" + dir.dir_path;
      new_name = sender_->name + "/" + name;
    }
  else
    {
      // create modified monitorable (w/ addition of sender's name)
      new_name = dir_path + ":";
      bool add_comma = false; // add comma between objects
      for(cvIt it = dir.contents.begin(); it != dir.contents.end(); ++it)
	{
	  if(add_comma)new_name += ",";
	  // add next monitoring element
	  new_name += *it;
	  add_comma = true;	
	}
    }

  // avoid using setCurrentFolder in case other thread is modifying fCurrentFolder;
  // use MonitorElementRootFolder methods instead
  ////  bei->setCurrentFolder(dir_path);


  for(vIt it = dir.contents.begin(); it != dir.contents.end(); ++it)
    {
      if(!bei->findObject(*it, dir_path))
	bei->addElement(*it, dir_path);
    }

  bei->lock();
  bei->addedMonitorable.push_back(new_name);
  bei->unlock();

  // keep this flag true until Sender sends "kMESS_DONE_MONITORABLE"
  sender_->need2readMonitorable = true;

  return true;
}

// remove objects from monitorable; "name" format: see DirFormat definition
// return success flag
bool ReceiverBase::removeMonitorable(const string & name)
{
  if(bei->DQM_VERBOSE)
    NodeBase::showMessage(" Monitorable removable from node ",
			  sender_->name, name);

  // unpack "name" into dir_path and contents
  DirFormat dir;
  if(!unpackDirFormat(name, dir))
    return false;

  // this is the real path for the monitorable directory
  string monit_path = sender_->name;
  if(!isTopFolder(dir.dir_path))
    monit_path += "/" + dir.dir_path;


  MonitorElementRootFolder * folder = bei->getDirectory(monit_path);
  if(!folder)
    {
      cerr << " *** Ignored command to remove directory " << monit_path 
	   << " from monitorable " << endl;
      return false;
    }

  for(vIt it = dir.contents.begin(); it != dir.contents.end(); ++it)
    {
      setCanDelete(folder, *it, true); // set canDelete to true
      // remove object
      bei->removeElement(folder, *it, false); // no warnings
    }
  // if contents: empty, remove all contents
  if(dir.contents.empty())
    {
      setCanDelete(folder, true); // set canDelete = true for all contents
      // remove all objects
      bei->removeContents(folder);
    }
  // new monitorable removal: add sender's name to new monitorable
  string new_name = sender_->name + "/" + name;

  bei->lock();
  bei->removedMonitorable.push_back(new_name);
  bei->unlock();

  // keep this flag true until Sender sends "kMESS_DONE_MONITORABLE"
  sender_->need2readMonitorable = true;

  return true;
}

// loop over bei->requests.toAdd, requests.toRemove: look for requests to sender_;
// when match found, update private members addMe, removeMe 
void ReceiverBase::produceSubscription(void)
{
  LockMutex a(bei->requests.mutex);

  lIt it; vector<lIt> matches;
  // addition requests
  for(it = bei->requests.toAdd.begin(); it != bei->requests.toAdd.end(); 
      ++it)
    {
      // look for request containing sender's name
      if(belongs2folder(sender_->name, *it))
	{
	  modifySubscription(*it, true);
	  matches.push_back(it);
	}
    }

  for(vector<lIt>::iterator it2 = matches.begin(); it2 != matches.end(); 
      ++it2)
    bei->requests.toAdd.erase(*it2);

  matches.clear();

  // removal requests
  for(it = bei->requests.toRemove.begin(); 
      it != bei->requests.toRemove.end(); ++it)
    {
      // look for request containing sender's name
      if(belongs2folder(sender_->name, *it))
	{
	  modifySubscription(*it, false);
	  matches.push_back(it);
	}
    }

  for(vector<lIt>::iterator it2 = matches.begin(); it2 != matches.end(); 
      ++it2)
    bei->requests.toRemove.erase(*it2);

  matches.clear();
}

 
// prepare subscription request; name format: see DirFormat definition
// if flag=true: add subscription, else: cancel subscription;
// non-existing MEs are discarded from <name>
void ReceiverBase::modifySubscription(string name, bool add)
{
  // First, make sure that the request makes sense

  // unpack "name" into dir_path and contents
  DirFormat dir;
  if(!unpackDirFormat(name, dir))
    return;

  string subdir;
  if(!isTopFolder(dir.dir_path))
    subdir = dir.dir_path;
  else
    subdir = ROOT_PATHNAME;

  MonitorElementRootFolder * folder = bei->getDirectory(subdir);
  if(!folder)
    {
      cerr << " *** Directory " << dir.dir_path << " does not exist!\n";
      cerr << " *** Ignored subscription request: <" << name << "> " << endl;
      return;
    }
  
  string fixedContents;
  // look for "bogus" monitoring elements; if any, discard
  bool foundBogus = false;
  for(vIt it = dir.contents.begin(); it != dir.contents.end(); ++it)
    {
      if(!folder->hasMonitorable(*it))
	{
	  cerr << " *** Directory " << dir.dir_path << " has no ME named "
	       << *it << endl;
	  cerr << " Subscription request ignored... " << endl;
	  foundBogus = true;
	  continue;
	}
      else
	{
	  if(fixedContents.empty())fixedContents += ",";
	  fixedContents += *it;
	}
    }
  // modify <name> if at least one bogus ME was found
  if(foundBogus)
    {
      // make sure the bogus ME was not the only one!
      if(!fixedContents.empty())
	name = dir.dir_path + ":" + fixedContents;
      else
	return; // invalid request, we are done here
    }


  for(vIt it = dir.contents.begin(); it != dir.contents.end(); ++it)
    {
      // set "desire" flag for ME
      setIsDesired(folder, *it, add);
      
      // remove from memory via bei;
      // no need to call book1D (etc) when subscribing; 
      // will be done once object has been sent
      if(!add)bei->removeElement(folder, *it);
    }
  
  if(dir.contents.empty())
    {
      // set "desire" flag for directory
      setIsDesired(folder, add);
      
      // if contents: empty, remove all contents from memory
      if(!add)bei->removeContents(folder);
      
    }

  // chop off sender's name before registering subscription,
  // e.g. FU0/C1/C2:histo --> C1/C2:histo,
  //        or  FU0:histo --> .:histo

  if(sender_->name == dir.dir_path)
    { // here we have something like "F0:histo"
      int offset = (sender_->name).size(); 
      // remove sender's name (eg. "FU0")
      string name_prime(name.c_str() + offset);
      // add pathname of root
      name = ROOT_PATHNAME + name_prime;
    }
  else
    { // here we have something like "F0/C1/C2:histo"
      int offset = (sender_->name).size() + 1;
      // remove sender's name + "slash" (eg. "FU0/")
      string name_prime(name.c_str() + offset);
      name = name_prime;
    }

  if(add)
    sender_->addMe.push_back(subscAddPrefix + name);
  else
    sender_->removeMe.push_back(subscRemPrefix + name);

}

// send subscription to sender
void ReceiverBase::sendSubscription(void)
{
  bool sentSomething = false;

  // first send new subscription requests
  for(vIt it = sender_->addMe.begin(); it != sender_->addMe.end(); ++it)
    {
      // receiver subscribing to all monitoring elements
      if(bei->DQM_VERBOSE)
	NodeBase::showMessage(" Sending subscription request to ", 
			      sender_->name, *it);
      int nerr = SocketUtils::sendString (it->c_str (), recv_socket);
      if(nerr <= 0)
	{
	  cout << " *** Error " << nerr;
	  NodeBase::showMessage(" while sending subscription request to ", 
				sender_->name, *it);
	}
      else
	sentSomething = true;
    }
  // then send subscription cancellations
  for(vIt it = sender_->removeMe.begin(); it != sender_->removeMe.end(); 
      ++it)
    {
      // receiver subscribing to all monitoring elements
      if(bei->DQM_VERBOSE)
	NodeBase::showMessage(" Sending subscription cancelation request to ", 
			      sender_->name, *it);

      int nerr = SocketUtils::sendString (it->c_str (), recv_socket);
      if(nerr <= 0)
	{
	  cout << " *** Error " << nerr;
	  NodeBase::showMessage(" while sending subscription cancelation request to ", sender_->name, *it);
	}
      else
	sentSomething = true;
    }

  // done sending suscription requests/cancellations, send "done" message to sender
  if(sentSomething)
    {
      DQMMessage mess1;
      mess1.setWhat (kMESS_DONE_SUBSCRIPTION);
      int nerr = SocketUtils::sendMessage (&mess1, recv_socket);
      if(nerr <= 0)
	cerr << " *** Error " << nerr 
	     << " while sending kMESS_DONE_SUBSCRIPTION to node "
	     << sender_->name << endl;
    }
  sender_->addMe.clear(); sender_->removeMe.clear();
}

// get object description sent by Sender; format: see checkObjDesc
void ReceiverBase::getObjectDesc(const string & name)
{
  vector<string> desc; unpackString(name.c_str(), ":", desc);
  if(!checkObjDesc(desc))
    {
      sender_->path = "unknown_path";
      sender_->objn = 0;
      return;
    }
  cvIt it = desc.begin();
  sender_->path = it->c_str(); ++it;
  sender_->objn = int(atoi(it->c_str()));

  if(sender_->objn > 0)
    sender_->need2readMonitoring = true;
}

// save output file
void ReceiverBase::saveFile(const string & filename)
{
  bei->save(filename);
}

// true if there are new (un)subscription requests
bool ReceiverBase::canSubscribe(void) const
{
  LockMutex a(bei->requests.mutex);
  return(!bei->requests.toAdd.empty() || !bei->requests.toRemove.empty());
}

// true if there is updated monitorable
bool ReceiverBase::newMonitorable(void) const
{
  return !bei->addedMonitorable.empty() ||  
    !bei->removedMonitorable.empty();
}

// remove directory <subdir> from monitorable
void ReceiverBase::lessMonitorable(const string subdir)
{
  // get list of removed contents; updated removed-monitorable list 
  // (will be sent to downstream class if available)
  vector<string> put_here;
  bei->getRemovedContents(put_here); 
  for(cvIt it = put_here.begin(); it != put_here.end(); ++it)
    {
      // look for request containing <subdir>
      if(belongs2folder(subdir, *it))
	bei->removedMonitorable.push_back(*it);
    } // loop over removed contents
}

// add directory <subdir> to monitorable
void ReceiverBase::moreMonitorable(const string subdir)
{
  // get list of added contents; updated added-monitorable list 
  // (will be sent to downstream class if available)
  vector<string> put_here;
  bei->getAddedContents(put_here);
  for(cvIt it = put_here.begin(); it != put_here.end(); ++it)
    {
      // look for request containing <subdir>
      if(belongs2folder(subdir, *it))
	bei->addedMonitorable.push_back(*it);
    } // loop over added contents
}

// set ME's "canDelete" property in directory <folder>; 
// to be used to set property to false when ME is extracted in this class
void ReceiverBase::setCanDelete(MonitorElementRootFolder * folder, 
				string ME_name, bool flag) const
{
  if(!folder)
    return;

  if(!folder->hasMonitorable(ME_name))
    {
      cerr << " *** Object " << ME_name << " does not exist in "
	   << folder->getPathname() << endl;
      cerr << " Ignoring setCanDelete operation... " << endl;
      return;
    }
  
  folder->canDeleteFromMenu[ME_name] = flag;
}

// call setCanDelete in current directory
//void ReceiverBase::setCanDelete(string ME_name, bool flag) const
//{
//  MonitorElementRootFolder * folder = (MonitorElementRootFolder *)
//    bei->getDirectory(bei->pwd());
//  setCanDelete(folder, ME_name, flag);
//}

// call setCanDelete for all ME in directory <folder>
void ReceiverBase::setCanDelete(MonitorElementRootFolder * folder, bool flag) const
{
  if(!folder)
    return;

  for(cME_it it = folder->objects_.begin(); it != folder->objects_.end(); 
      ++it)
    setCanDelete(folder, it->first, flag);

}
// set <name> ME's "isDesired" property in <folder>: to be used 
// to set property to true/false when ME is (un)subscribed in ReceiverBase class
void ReceiverBase::setIsDesired(MonitorElementRootFolder * folder, 
				string ME_name, bool flag) const
{
  if(!folder)
    {
      cerr << " *** Null MonitorElementRootFolder in ReceiverBase::setIsDesired"
	   << endl;
      return;
    }

  if(!folder->hasMonitorable(ME_name))
    {
      cerr << " *** Object " << ME_name << " does not exist in "
	   << folder->getPathname() << endl;
      cerr << " Ignoring setIsDesired operation... " << endl;
      return;
    }
  
  folder->isDesired[ME_name] = flag;
}

// call setIsDesired for all ME in <dir>
void ReceiverBase::setIsDesired(MonitorElementRootFolder* folder, bool flag) 
  const
{
  if(!folder)
    {
      cerr << " *** Null MonitorElementRootFolder in ReceiverBase::setIsDesired"
	   << endl;
      return;
    }

  for(cME_it it = folder->objects_.begin(); it != folder->objects_.end(); 
      ++it)
    setIsDesired(folder, it->first, flag);
}

// true if Monitoring Element <me> in directory <folder> has isDesired = true;
// if warning = true and <me> does not exist, show warning
bool ReceiverBase::isDesired(MonitorElementRootFolder * folder, string me, 
			     bool warning) const
{
  if(!folder)
    return false;

  return bei->isDesired(folder, me, warning);
}

// true if sender is done sending (a) monitorable & (b) monitoring
bool ReceiverBase::isSenderDone(void) const
{
  return !sender_->need2readMonitorable && 
    !sender_->need2readMonitoring;
}

// add tags to (addFlag=true) or remove tags from (addFlag = false) monitorable; 
// "name" format: see DirFormat definition
// (with exception that <obj> is replaced by <obj>/tag1/tag2, etc.
// return success flag
bool ReceiverBase::modifyTags(const string & name, bool addFlag)
{
  if(bei->DQM_VERBOSE)
    {
      if(addFlag)
	cout << " Addition";
      else
	cout << " Removal";
      cout << " of tags from node " << sender_->name 
	   << ": <" << name << "> " << endl;
    }
  // unpack "name" into dir_path and contents (WITH TAGS)
  DirFormat dir;
  if(!unpackDirFormat(name, dir))
    return false;

  string dir_path = sender_->name;
  // add subdirectory (unless it is the top folder)
  if(!isTopFolder(dir.dir_path))
    dir_path += "/" + dir.dir_path;

  MonitorElementRootFolder * folder = bei->getDirectory(dir_path);
  
  for(cvIt it = dir.contents.begin(); it != dir.contents.end(); ++it)
    {// loop over all MEs (with tags)
     
      vector<string> data;
      // ME name and tags are separated by slash ("/")
      unpackString((*it).c_str(), "/", data);
      
      bool firstItem = true;
      string ME_name; unsigned int tag_no;
      MonitorElement * me = 0;
      for(cvIt it2 = data.begin(); it2 != data.end(); ++it2)
	{ // loop over ME and its tags
	  if(firstItem) // first item in data is ME_name;
	    {
	      ME_name = *it2;
	      if(folder)
		me = folder->findObject(ME_name);

	      firstItem = false;
	    }
	  else
	    {
	      // converting int to "unsigned int"! is there a better way?
	      tag_no = (unsigned int) atoi((*it2).c_str());
	      if(me)
		{
		  // ME is already present (tag arrives after <me>)
		  if(addFlag)
		    bei->tag(me, tag_no);
		  else
		    bei->untag(me, tag_no);
		}
	      else
		{
		  // ME is not present (tag arrived before <me>)
		  if(addFlag)
		    bei->tagHelper->tag(dir_path, ME_name, tag_no);
		  else
		    bei->tagHelper->untag(dir_path, ME_name, tag_no);
		}
	    }
	} // loop over ME and its tags
    }// loop over all MEs (with tags)
  
  // keep this flag true until Sender sends "kMESS_DONE_MONITORABLE"
  sender_->need2readMonitorable = true;

  return true;
}
