#ifndef MonitorUserInterface_h
#define MonitorUserInterface_h

class MonitorElement;
class CollateMonitorElement;

#include "DQMServices/Core/interface/StringUtil.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Core/interface/QCriterion.h"

#include <list>
#include <map>

class MonitorUserInterface : public StringUtil
{

 public:
  
  MonitorUserInterface(const std::string & hostname, int port_no, 
		       const std::string & client_name);
  virtual ~MonitorUserInterface();
  
  // ---------------- Getters -----------------------------
  // get ME from full pathname (e.g. "my/long/dir/my_histo")
  virtual MonitorElement * get(const std::string & fullpath) const = 0;
  // this is the "main" loop where we receive monitoring;
  // returns success flag
  bool update(void);

  // *****************  NOTE ******************************************
  // instead of calling "update", users can use the following two methods:
  // 1. retrieval of monitoring, sending of subscription requests/cancellations,
  // calculation of "collate"-type Monitoring Elements;
  // returns success flag
  virtual bool doMonitoring(void) = 0;
  // 2. run quality tests on all MonitorElements that have been updated (or added)
  // since last monitoring cycle
  void runQTests(void)
  {bei->runQTests();}
  // *****************  NOTE ******************************************

  // return # of monitoring cycles received
  virtual int getNumUpdates(void) const = 0;

  // get pointer to back-end interface
  DaqMonitorBEInterface * getBEInterface(void){return bei;}

  // get all contents
  // return vector<string> of the form <dir pathname>:<obj1>,<obj2>,<obj3>
  void getContents(std::vector<std::string> & put_here) const
  {bei->getContents(put_here);}
  // get monitorable
  // return vector<string> of the form <dir pathname>:<obj1>,<obj2>,<obj3>
  void getMonitorable(std::vector<std::string> & put_here) const
  {bei->getMonitorable(put_here);}
  // get added monitorable (since last cycle)
  // return vector<string> of the form: <dir pathname>:<obj1>,<obj2>,...
  void getAddedMonitorable(std::vector<std::string> & put_here) const
  {bei->getAddedMonitorable(put_here);}
  // get removed monitorable (since last cycle)
  // return vector<string> of the form: <dir pathname>:<obj1>,<obj2>,...
  void getRemovedMonitorable(std::vector<std::string> & put_here) const
  {bei->getRemovedMonitorable(put_here);}
  // get added contents (since last cycle)
  // return vector<string> of the form: <dir pathname>:<obj1>,<obj2>,...
  void getAddedContents(std::vector<std::string> & put_here) const
  {bei->getAddedContents(put_here);}
  // get removed contents (since last cycle)
  // return vector<string> of the form: <dir pathname>:<obj1>,<obj2>,...
  void getRemovedContents(std::vector<std::string> & put_here) const
  {bei->getRemovedContents(put_here);}
  // get updated contents (since last cycle)
  // *** included added content!
  // return vector<string> of the form: <dir pathname>:<obj1>,<obj2>,...
  void getUpdatedContents(std::vector<std::string> & put_here) const
  {bei->getUpdatedContents(put_here);}
  // ---------------- Browsing -----------------------------
  // get list of subdirectories of current directory
  std::vector<std::string> getSubdirs(void) const
    {return bei->getSubdirs();}
  // get list of (non-dir) MEs of current directory (includes null MEs)
  std::vector<std::string> getMEs(void) const
    {return bei->getMEs();}
  // retun pathname of current directory
  std::string pwd(void) const
    {return bei->pwd();}
  // go to top directory (ie. root)
  void cd(void) const
  {bei->cd();}
  // equivalent to "cd .."
  void goUp(void)
  {bei->goUp();}
  // set the last directory in fullpath as the current directory (create if needed)
  // to be invoked by user to specify directories for monitoring objects b4 booking
  // commands book1D (etc) & removeElement(name) imply elements in this directory!
  void setCurrentFolder(std::string fullpath)
  {bei->setCurrentFolder(fullpath);}
  // cd to subdirectory (if there)
  void cd(std::string subdir_path)
  {bei->cd(subdir_path);}

  // true if directory (or any subfolder at any level below it) contains
  // at least one valid (i.e. non-null) monitoring element
  bool containsAnyMEs(const std::string & pathname) const
  {return bei->containsAnyMEs(pathname);}
  // true if directory (or any subfolder at any level below it) contains
  // at least one monitorable element
  bool containsAnyMonitorable(const std::string & pathname) const
  {return bei->containsAnyMonitorable(pathname);}

  // ---------------- Subscriptions -----------------------------
  
  // subscription request; format: (a) exact pathname (e.g. A/B/C/histo)
  // (b) or with wildcards (e.g. A/?/C/histo, A/B/*/histo or A/B/*)
  void subscribe(const std::string & subsc_request);
  // unsubscription request; format: (a) exact pathname (e.g. A/B/C/histo)
  // (b) or with wildcards (e.g. A/?/C/histo, A/B/*/histo or A/B/*)
  void unsubscribe(const std::string & subsc_request);
  // similar to method subscribe; use only additions to monitorable in last cycle
  void subscribeNew(const std::string & subsc_request);

  // ---------------- Miscellaneous -----------------------------
  
   // save structure with monitoring objects into root file
  void save(const std::string & filename);
  // cycle through all monitoring objects, draw one at time
  void drawAll(void) const;
  // attempt to connect (to be used if ctor failed to connect)
  virtual void connect(std::string host, unsigned port) = 0;
  // opposite action of that of connect method
  virtual void disconnect(void) = 0; 
  // true if connection was succesful
  virtual bool isConnected(void) const = 0;
  // set reconnect delay parameter (in secs);
  // use delay < 0 for no reconnection attempts
  virtual void setReconnectDelay(int delay) = 0;
  // set verbose level (0 turns all non-error messages off)
  void setVerbose(unsigned level) const
  {bei->setVerbose(level);}

  // ------------ Operations for MEs that are normally never reset ---------

  // reset contents (does not erase contents permanently)
  // (makes copy of current contents; will be subtracted from future contents)
  void softReset(MonitorElement * me);

  // if true: will subtract contents copied at "soft-reset" from now on
  // if false: will NO longer subtract contents (default)
  // Note: after enabling, users much call "softReset" to reset ME
  void enableSoftReset(MonitorElement * me, bool flag);

  // --- Operations on MEs that are normally reset at end of monitoring cycle ---

  // if true, will accumulate ME contents (over many periods)
  // until method is called with flag = false again
  void setAccumulate(MonitorElement * me, bool flag);

  // --------------------- Collation of MEs --------------------------------
  // collate 1D histograms, store in <pathname>
  virtual CollateMonitorElement * collate1D(const std::string name, 
					    const std::string title, 
					    const std::string pathname)=0;
  // collate 2D histograms, store in <pathname>
  virtual CollateMonitorElement * collate2D(const std::string name, 
					    const std::string title, 
					    const std::string pathname)=0;
  // collate 3D histograms, store in <pathname>
  virtual CollateMonitorElement * collate3D(const std::string name, 
					    const std::string title, 
					    const std::string pathname)=0;
  // collate profiles, store in <pathname>
  virtual CollateMonitorElement* collateProf(const std::string name, 
					     const std::string title, 
					     const std::string pathname)=0;
  
  // collate profiles, store in <pathname>
  virtual CollateMonitorElement* collateProf2D(const std::string name, 
					       const std::string title, 
					       const std::string pathname)
    =0;
  
  // add <search_string> to summary ME; 
  // <search_string> could : (a) be exact pathname (e.g. A/B/C/histo)
  // (b) include wildcards (e.g. A/?/C/histo, A/B/*/histo or A/B/*);
  // this action applies to all MEs already available or future ones
  void add(CollateMonitorElement* cme, const std::string& search_string) const;


  // -------------------- Quality tests on MonitorElements ------------------

  // create quality test with unique name <qtname> (analogous to ME name);
  // quality test can then be attached to ME with useQTest method
  // (<algo_name> must match one of known algorithms)
  QCriterion * createQTest(std::string algo_name, std::string qtname)
  {return bei->createQTest(algo_name, qtname);}
  
  // attach quality test <qtname> to all ME matching <search_string>;
  // <search_string> could : (a) be exact pathname (e.g. A/B/C/histo)
  // (b) include wildcards (e.g. A/?/C/histo, A/B/*/histo or A/B/*);
  // this action applies to all MEs already available or future ones
  void useQTest(std::string search_string, std::string qtname) const
  {bei->useQTest(search_string, qtname);}
  // get quality test with name <qtname> (null pointer if no such test)
  QCriterion * getQCriterion(std::string qtname) const
  {return bei->getQCriterion(qtname);}
  
 protected:

  // do calculations for all collate MEs; come here at end of monitoring cycle)
  void doSummary(void);

  // to be used by methods subscribe (add=true) and unsubscribe (add=false)
  // <subsc_request> format: (a) exact pathname (e.g. A/B/C/histo)
  // (b) or with wildcards (e.g. A/?/C/histo, A/B/*/histo or A/B/*)
  void subscribe_base(const std::string & subsc_request, bool add);
  
  // set(unset) subscription if add=true(false)
  void finishSubscription(const std::vector<std::string> & monit, bool add);
  
  // collector name (e.g. <my_collector> or <collector1/collector2>
  std::string collector_name_;

  // vector of collation MEs
  std::vector<CollateMonitorElement *> collate_mes;
  // new MEs have been added; check if need to update collate-MEs
  void checkAddedContents(void);
  
 // use to get hold of structure with monitoring elements that class owns
  DaqMonitorBEInterface *bei;
  // # of monitoring packages received;
  int numUpdates_;
  
};


#endif
