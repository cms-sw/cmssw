#ifndef MonitorUserInterface_h
#define MonitorUserInterface_h

class MonitorElement;
class CollateMonitorElement;

#include "DQMServices/Core/interface/StringUtil.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Core/interface/QCriterion.h"

#include <SealBase/Callback.h>

#include <list>
#include <map>

class MonitorElementRootFolder;

class MonitorUserInterface : public StringUtil
{

 protected:
  /** Connect with monitoring server (DQM Collector) at <hostname> and <port_no>
     using <client_name>; if flag=true, client will accept downstream connections
     MonitorUserInterface(std::string hostname,int port_no,std::string client_name,

     int reconnect_delay_secs = 5, bool actAsServer = false); */

  /** Connect with monitoring server (DQM Collector) with a list of hostnames at 
     <port_no> using <client_name>;   
     if flag=true, client will accept downstream connections
     MonitorUIRoot::MonitorUIRoot(std::vector<std::string> hostnames, int port_no, 
     std::string client_name, int reconnect_delay_secs=5, bool actAsServer=false); 
  */ 

  /** Use the default constructor for running in standalone mode (ie. without
     sources or collectors); if flag=true, client will accept downstream connections
  */
  MonitorUserInterface();
 
  /// when in "standalone mode", there are no upstream connections
  /// (but there may be downstream clients...)
  bool needUpstreamConnections() const {return !standaloneMode_;}
  // 
  bool standaloneMode_;


 public:

  virtual ~MonitorUserInterface();
  

  /// add call back; to be used for thread-unsafe operations
  virtual void addCallback(seal::Callback & action) = 0;

  // ---------------- Getters -----------------------------
  /// get ME from full pathname (e.g. "my/long/dir/my_histo")
//  MonitorElement * get(const std::string & fullpath) const
//  {return bei->get(fullpath);}
  /** this is the "main" loop where we receive monitoring or
      send subscription requests;
      if client acts as server, method runQTests is also sending monitoring & 
      test results to clients downstream;
      returns success flag */
  bool update(void);

  // *****************  NOTE ******************************************
  /** instead of calling "update", users can use the following two methods:
      1. Retrieval of monitoring, sending of subscription requests/cancellations,
      calculation of "collate"-type Monitoring Elements;
      returns success flag */
  virtual bool doMonitoring(void) = 0;
  /** 2. Run quality tests on all MonitorElements that have been updated (or added)
      since last monitoring cycle;
      Method is overloaded if client acts as server to other clients downstream */
//  virtual void runQTests(void)
//  {bei->runQTests();}
  // *****************  NOTE ******************************************

  /// allow downstream clients to connect to this client
  /// (to be used only with no-arg ctor; use boolean switch for other ctors)
  virtual void actAsServer(int port_no, std::string client_name) = 0;
  /// true if client accepts downstream connections/requests
  virtual bool isServer() const = 0;

  /// return # of monitoring cycles received
  virtual int getNumUpdates(void) const = 0;
  /// set maximum # of attempts to reconnect to server (upon connection problems)
  virtual void setMaxAttempts2Reconnect(unsigned Nrecon_attempts) = 0;

  /// get pointer to back-end interface
  DaqMonitorBEInterface * getBEInterface(void){return bei;}

  /// get all contents
  /// return vector<string> of the form <dir pathname>:<obj1>,<obj2>,<obj3>
//  void getContents(std::vector<std::string> & put_here) const
//  {bei->getContents(put_here);}
  /// get monitorable
  /// return vector<string> of the form <dir pathname>:<obj1>,<obj2>,<obj3>
//  void getMonitorable(std::vector<std::string> & put_here) const
//  {bei->getMonitorable(put_here);}
  /// get added monitorable (since last cycle)
  /// return vector<string> of the form: <dir pathname>:<obj1>,<obj2>,...
//  void getAddedMonitorable(std::vector<std::string> & put_here) const
//  {bei->getAddedMonitorable(put_here);}
  /// get removed monitorable (since last cycle)
  /// return vector<string> of the form: <dir pathname>:<obj1>,<obj2>,...
//  void getRemovedMonitorable(std::vector<std::string> & put_here) const
//  {bei->getRemovedMonitorable(put_here);}
  /// get added contents (since last cycle)
  /// return vector<string> of the form: <dir pathname>:<obj1>,<obj2>,...
//  void getAddedContents(std::vector<std::string> & put_here) const
//  {bei->getAddedContents(put_here);}
  /// get removed contents (since last cycle)
  /// return vector<string> of the form: <dir pathname>:<obj1>,<obj2>,...
//  void getRemovedContents(std::vector<std::string> & put_here) const
//  {bei->getRemovedContents(put_here);}
  /// get updated contents (since last cycle)
  /// *** included added content!
  /// return vector<string> of the form: <dir pathname>:<obj1>,<obj2>,...
//  void getUpdatedContents(std::vector<std::string> & put_here) const
//  {bei->getUpdatedContents(put_here);}
  // ---------------- Browsing -----------------------------
  /// get list of subdirectories of current directory
//  std::vector<std::string> getSubdirs(void) const
//    {return bei->getSubdirs();}
  /// get list of (non-dir) MEs of current directory (includes null MEs)
//  std::vector<std::string> getMEs(void) const
//   {return bei->getMEs();}
  /// retun pathname of current directory
//  std::string pwd(void) const
//    {return bei->pwd();}
  /// go to top directory (ie. root)
//  void cd(void) const
//  {bei->cd();}
  /// equivalent to "cd .."
//  void goUp(void)
//  {bei->goUp();}
  /// set the last directory in fullpath as the current directory (create if needed)
  /// to be invoked by user to specify directories for monitoring objects b4 booking
  /// commands book1D (etc) & removeElement(name) imply elements in this directory!
//  void setCurrentFolder(std::string fullpath)
//  {bei->setCurrentFolder(fullpath);}
  /// cd to subdirectory (if there)
//  void cd(std::string subdir_path)
//  {bei->cd(subdir_path);}

  /// true if directory (or any subfolder at any level below it) contains
  /// at least one valid (i.e. non-null) monitoring element
//  bool containsAnyMEs(const std::string & pathname) const
//  {return bei->containsAnyMEs(pathname);}
  /// true if directory (or any subfolder at any level below it) contains
  /// at least one monitorable element
//  bool containsAnyMonitorable(const std::string & pathname) const
//  {return bei->containsAnyMonitorable(pathname);}

  // ---------------- Subscriptions -----------------------------
  
  /// subscription request; format: 
  /// (a) exact pathname with ME name (e.g. A/B/C/histo) ==> FAST
  /// (b) or with wildcards (e.g. A/?/C/histo, A/B/*/histo or A/B/*) ==> SLOW
  void subscribe(std::string subsc_request);
  /// subscription request for directory contents ==> FAST
  /// (need exact pathname without wildcards, e.g. A/B/C)
  /// use flag to specify whether subfolders (& their contents) should be included;
  /// Users are encourage to use this method instead of previous one w/ wildcards
  void subscribe(std::string subsc_request, bool useSubFolders);
  
  /// same as above for tagged MonitorElements
  void subscribe(std::string subsc_request, unsigned int tag);
  void subscribe(std::string subsc_request, bool useSubFolders, 
		 unsigned int tag);
  /// subscription request for all MEs with given tag ==> FAST
  void subscribe(unsigned int tag);

  /// similar to above subscription methods; 
  /// use only additions to monitorable in last cycle
  void subscribeNew(std::string subsc_request);
  void subscribeNew(std::string subsc_request, bool useSubFolders); 
  void subscribeNew(std::string subsc_request, unsigned int tag);
  void subscribeNew(std::string subsc_request, bool useSubFolders,
		    unsigned int tag);
  void subscribeNew(unsigned int tag);

  /// unsubscription requests; format is same with subscription requests 
  void unsubscribe(std::string subsc_request);
  void unsubscribe(std::string subsc_request, bool useSubFolders);
  void unsubscribe(std::string subsc_request, unsigned int tag);
  void unsubscribe(std::string subsc_request, bool useSubFolders,
		   unsigned int tag);
  ///
  void unsubscribe(unsigned int tag);

  // ---------------- Miscellaneous -----------------------------
  
  /// save dir_fullpath with monitoring objects into root file <filename>;
  /// include quality test results with status >= minimum_status 
  /// (defined in Core/interface/QTestStatus.h);
  /// if dir_fullpath="", save full monitoring structure
//  void save(std::string filename, std::string dir_fullpath="",
//	    int minimum_status=dqm::qstatus::STATUS_OK) const;
  /// cycle through all monitoring objects, draw one at time
//  void drawAll(void) const;
  /// attempt to connect (to be used if ctor failed to connect)
  virtual void connect(std::string host, unsigned port) = 0;
  /// opposite action of that of connect method
  virtual void disconnect(void) = 0; 
  /// true if connection was succesful
  virtual bool isConnected(void) const = 0;
  /// set reconnect delay parameter (in secs);
  /// use delay < 0 for no reconnection attempts
  virtual void setReconnectDelay(int delay) = 0;
  /// set verbose level (0 turns all non-error messages off)
//  void setVerbose(unsigned level) const
//  {bei->setVerbose(level);}

  // ------------ Operations for MEs that are normally never reset ---------

  /// reset contents (does not erase contents permanently)
  /// (makes copy of current contents; will be subtracted from future contents)
//  void softReset(MonitorElement * me);

  /// reverts action of softReset
//  void disableSoftReset(MonitorElement * me);

  // --- Operations on MEs that are normally reset at end of monitoring cycle ---

  /// if true, will accumulate ME contents (over many periods)
  /// until method is called with flag = false again
//  void setAccumulate(MonitorElement * me, bool flag);

  // --------------------- Collation of MEs --------------------------------
  /// collate 1D histograms, store in <pathname>
  virtual CollateMonitorElement * collate1D(const std::string name, 
					    const std::string title, 
					    const std::string pathname)=0;
  /// collate 2D histograms, store in <pathname>
  virtual CollateMonitorElement * collate2D(const std::string name, 
					    const std::string title, 
					    const std::string pathname)=0;
  /// collate 3D histograms, store in <pathname>
  virtual CollateMonitorElement * collate3D(const std::string name, 
					    const std::string title, 
					    const std::string pathname)=0;
  /// collate profiles, store in <pathname>;
  virtual CollateMonitorElement* collateProf(const std::string name, 
					     const std::string title, 
					     const std::string pathname)=0;
  
  /// collate profiles, store in <pathname>;
  virtual CollateMonitorElement* collateProf2D(const std::string name, 
					       const std::string title, 
					       const std::string pathname)
    = 0;
  
  /// add <search_string> to summary ME; 
  /// <search_string> could : (a) be exact pathname (e.g. A/B/C/histo): FAST
  /// (b) include wildcards (e.g. A/?/C/histo, A/B/*/histo or A/B/*): SLOW
  /// this action applies to all MEs already available or future ones
  void add(CollateMonitorElement * cme, std::string search_string) const;
  /// same as above for tagged MEs
  void add(CollateMonitorElement * cme, unsigned int tag, 
	   std::string search_string) const;
  /// add directory contents to summary ME ==> FAST
  /// (need exact pathname without wildcards, e.g. A/B/C);
  /// use flag to specify whether subfolders (& their contents) should be included;
  /// this action applies to all MEs already available or future ones
  void add(CollateMonitorElement* cme, std::string pathname, bool useSubfolds)
    const;
  /// same as above for tagged MEs
  void add(CollateMonitorElement* cme,unsigned int tag,std::string pathname,
	   bool useSubfolds) const;
  /// add tagged MEs to summary ME ==> FAST
  /// this action applies to all MEs already available or future ones
  void add(CollateMonitorElement * cme, unsigned int tag) const;
  ///
  /// remove CollateMonitorElement
  void removeCollate(CollateMonitorElement * cme);

  // -------------------- Quality tests on MonitorElements ------------------

  /// create quality test with unique name <qtname> (analogous to ME name);
  /// quality test can then be attached to ME with useQTest method
  /// (<algo_name> must match one of known algorithms)
//  QCriterion * createQTest(std::string algo_name, std::string qtname)
//  {return bei->createQTest(algo_name, qtname);}
  
  /// attach quality test <qtname> to all ME matching <search_string>;
  /// <search_string> could : (a) be exact pathname (e.g. A/B/C/histo): FAST
  /// (b) include wildcards (e.g. A/?/C/histo, A/B/*/histo or A/B/*): SLOW
  /// this action applies to all MEs already available or future ones
//  void useQTest(std::string search_string, std::string qtname) const;
  /// same as above for tagged MEs
//  void useQTest(unsigned int tag, std::string search_string, 
//		std::string qtname) const;
  /// attach quality test <qtname> to directory contents ==> FAST
  /// (need exact pathname without wildcards, e.g. A/B/C);
  /// use flag to specify whether subfolders (& their contents) should be included;
  /// this action applies to all MEs already available or future ones
//  void useQTest(std::string pathname, bool useSubfolds, std::string qtname) 
//    const;
  /// same as above for tagged MEs
//  void useQTest(unsigned int tag, std::string pathname, bool useSubfolds,
//		std::string qtname) const;
  /// attach quality test <qtname> to tagged MEs ==> FAST
  /// this action applies to all MEs already available or future ones
//  void useQTest(unsigned int tag, std::string qtname) const;

  /// get quality test with name <qtname> (null pointer if no such test)
//  QCriterion * getQCriterion(std::string qtname) const
//  {return bei->getQCriterion(qtname);}

  /// get "global" system status (one of: STATUS_OK, WARNING, ERROR, OTHER);
  /// returns most sever error, where ERROR > WARNING > OTHER > STATUS_OK;
  /// see Core/interface/QTestStatus.h for details on "OTHER" 
//  int getSystemStatus(void) const{return bei->getStatus();}
  /// same as above for any pathname
//  int getStatus(std::string pathname) const{return bei->getStatus(pathname);}
  /// same as above for a tag
//  int getStatus(unsigned int tag) const{return bei->getStatus(tag);}
  /// same as above for vector with MonitorElements
//  int getStatus(std::vector<MonitorElement *> & ME_group) const
//  {return bei->getStatus(ME_group);}

 protected:

  /// do calculations for all collate MEs; come here at end of monitoring cycle)
  void doSummary(void);

  /// to be used by methods subscribe (add=true) and unsubscribe (add=false)
  /// <subsc_request> format: (a) exact pathname (e.g. A/B/C/histo)
  /// (b) or with wildcards (e.g. A/?/C/histo, A/B/*/histo or A/B/*)
  void subscribe_base(const std::string & subsc_request, bool add,
		      const dqm::me_util::rootDir & Dir);
  
  /// (un)subscription request for directory contents ==> FAST
  /// (need exact pathname without wildcards, e.g. A/B/C);
  /// use flag to specify whether subfolders (& their contents) should be included;
  /// use add=true(false) to (un)subscribe
  virtual void subscribeDir(std::string & subsc_request, 
			    bool useSubFolders,
			    unsigned int myTag, bool add) = 0;

  /// same as above for MonitorElementRootFolder
  virtual void subscribeDir(MonitorElementRootFolder * folder, 
			    bool useSubFolders,
			    unsigned int myTag, bool add) = 0;

  /// look for MEs matching subsc_request with <tag> in <path>
  void subscribeNew(const std::string & subsc_request, unsigned int tag,
		    std::vector<std::string> & requests,
		    const dqm::me_util::cdirt_it & path);

  /// get all MEs with <tag> in <path>
  void subscribeNew(unsigned int tag, std::vector<std::string> & requests,
		    const dqm::me_util::cdirt_it & path);

  /// set(unset) subscription if add=true(false)
  void finishSubscription(const std::vector<std::string> & monit, bool add);
  
  /// collector name (e.g. <my_collector> or <collector1/collector2>
  std::string collector_name_;

  /// new MEs have been added; check if need to update collate-MEs
  void checkAddedContents(void);
  
 /// use to get hold of structure with monitoring elements that class owns
  DaqMonitorBEInterface *bei;
  /// # of monitoring packages received;
  int numUpdates_;
  
 private:
  /// like subscribe_base above, for one folder only
  virtual void subscribe_base
    (const std::string & subsc_request, bool add, 
     std::vector<std::string> & requests, 
     const MonitorElementRootFolder * folder) = 0;
  
};

#endif
