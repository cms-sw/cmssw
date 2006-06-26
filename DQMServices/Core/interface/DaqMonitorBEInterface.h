#ifndef DaqMonitorBEInterface_h
#define DaqMonitorBEInterface_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/StringUtil.h"
#include "DQMServices/Core/interface/QCriterion.h"
#include "DQMServices/Core/interface/QTestStatus.h"

#include <pthread.h>
#include <semaphore.h>

#include <iostream>
#include <vector>
#include <list>
#include <map>
#include <string>

class QCriterion;

class DaqMonitorBEInterface: public StringUtil
{

 public:
  
  DaqMonitorBEInterface(edm::ParameterSet const &pset)
  {
    pthread_mutex_init(&mutex_,0); DQM_VERBOSE = 1;
    resetStuff();
  }  
  virtual ~DaqMonitorBEInterface();
 
  void reParseConfig(const edm::ParameterSet &pset){}
 
  // ---------------------- Booking ------------------------------------
  // book 1D histogram
  virtual MonitorElement * book1D(std::string name, 
				  std::string title, 
				  int nchX, double lowX, double highX)=0;
  // book 2D histogram
  virtual MonitorElement * book2D(std::string name, 
				  std::string title, 
				  int nchX, double lowX, double highX, 
				  int nchY, double lowY, double highY)=0;
  // book 3D histogram
  virtual MonitorElement * book3D(std::string name, 
				  std::string title, 
				  int nchX, double lowX, double highX, 
				  int nchY, double lowY, double highY, 
				  int nchZ, double lowZ, double highZ)=0;
  // book profile; 
  // option is one of: " ", "s" (default), "i", "G" (see TProfile::BuildOptions)
  // (in a profile plot the number of channels in Y is disregarded)
  virtual MonitorElement * bookProfile(std::string name, 
				       std::string title, 
				       int nchX, double lowX, double highX, 
				       int nchY, double lowY,double highY,
				       char * option = "s")=0;
  // book 2-D profile;
  // option is one of: " ", "s" (default), "i", "G" (see TProfile2D::BuildOptions)
  // (in a 2-D profile plot the number of channels in Z is disregarded)
  virtual MonitorElement * bookProfile2D(std::string name, 
					 std::string title, 
					 int nchX, double lowX, double highX,
					 int nchY, double lowY,double highY,
					 int nchZ, double lowZ,double highZ,
					 char * option = "s") = 0;
  // book float
  virtual MonitorElement * bookFloat(std::string ) = 0;
  // book int
  virtual MonitorElement * bookInt(std::string ) = 0;
  // book string
  virtual MonitorElement * bookString(std::string, std::string )=0;
  
  // ---------------- Navigation -----------------------
  
  // retun pathname of current directory
  virtual std::string pwd(void) const = 0;
  // go to top directory (ie. root)
  virtual void cd(void) = 0;
  // equivalent to "cd .."
  virtual void goUp(void) = 0;
  // set the last directory in fullpath as the current directory (create if needed)
  // to be invoked by user to specify directories for monitoring objects b4 booking
  // commands book1D (etc) & removeElement(name) imply elements in this directory!
  virtual void setCurrentFolder(std::string fullpath)=0;
  // cd to subdirectory (if there)
  virtual void cd(std::string subdir_path) = 0;
  // name of global monitoring folder (containing all sources subdirectories)
  static const std::string monitorDirName;
  // name of global subscriber folder (containing all clients subdirectories)
  static const std::string subscriberDirName;
  // ---------------- Miscellaneous -----------------------------
  
  // show directory structure
  virtual void showDirStructure(void) const = 0;
  // save directory with monitoring objects into root file <filename>;
  // include quality test results with status >= minimum_status 
  // (defined in Core/interface/QTestStatus.h);
  // if directory="", save full monitoring structure
  virtual void save(std::string filename, std::string directory="",
		    int minimum_status=dqm::qstatus::STATUS_OK) = 0;
  // cycle through all monitoring objects, draw one at time
  virtual void drawAll(void) = 0;
  // get list of subdirectories of current directory
  virtual std::vector<std::string> getSubdirs(void) const = 0;
  // get list of (non-dir) MEs of current directory
  virtual std::vector<std::string> getMEs(void) const = 0;
  // set verbose level (0 turns all non-error messages off)
  void setVerbose(unsigned level){DQM_VERBOSE = level;}
  // get verbose level
  unsigned getVerbose(void) const {return DQM_VERBOSE;}

  // unpack QReport (with name, value) into ME_name, qtest_name, status, message;
  // return success flag; Expected format of QReport is a TNamed variable with
  // (a) name in the form: <ME_name>.<QTest_name>
  // (b) title (value) in the form: st.<status>.<the message here>
  // (where <status> is defined in Core/interface/QTestStatus.h)
  bool unpackQReport(std::string name, std::string value, 
		     std::string ME_name, std::string qtest_name,
		     int & status, std::string message) const
  {return StringUtil::unpackQReport(name, value, ME_name, qtest_name,
				    status, message);}
  
  // -------------------- Deleting ----------------------------------
  
  // remove directory
  virtual void rmdir(std::string fullpath) = 0;
  // erase monitoring element in current directory 
  // (opposite of book1D,2D,etc. action);
  virtual void removeElement(std::string name) = 0;
  // erase all monitoring elements in current directory (not including subfolders);
  virtual void removeContents(void) = 0;
  
  // acquire and release lock
  void lock();
  void unlock();
  
 protected:
  
  // ------------------- Private "getters" ------------------------------
  
  // get all contents;
  // return vector<string> of the form <dir pathname>:<obj1>,<obj2>,<obj3>;
  // if showContents = false, change form to <dir pathname>:
  // (useful for subscription requests; meant to imply "all contents")
  virtual void getContents(std::vector<std::string> & put_here,
			   bool showContents = true) const = 0;
  // get monitorable;
  // return vector<string> of the form <dir pathname>:<obj1>,<obj2>,<obj3>;
  // if showContents = false, change form to <dir pathname>:
  // (useful for subscription requests; meant to imply "all contents")
  virtual void getMonitorable(std::vector<std::string> & put_here,
			      bool showContents = true) const = 0;
  // get added monitorable (since last cycle)
  // return vector<string> of the form: <dir pathname>:<obj1>,<obj2>,...
  void getAddedMonitorable(std::vector<std::string> & put_here) const;
  // get removed monitorable (since last cycle)
  // return vector<string> of the form: <dir pathname>:<obj1>,<obj2>,...
  void getRemovedMonitorable(std::vector<std::string> & put_here) const;
  // get added contents (since last cycle)
  // return vector<string> of the form: <dir pathname>:<obj1>,<obj2>,...
  void getAddedContents(std::vector<std::string> & put_here) const;
  // get removed contents (since last cycle)
  // return vector<string> of the form: <dir pathname>:<obj1>,<obj2>,...
  void getRemovedContents(std::vector<std::string> & put_here) const;
  // get updated contents (since last cycle)
  // COMPLEMENTARY to addedContents, removedContents
  // return vector<string> of the form: <dir pathname>:<obj1>,<obj2>,...
  void getUpdatedContents(std::vector<std::string> & put_here) const;
 
  // get folder corresponding to inpath wrt to root (create subdirs if necessary)
  virtual MonitorElement * makeDirectory(std::string inpath) = 0;
  // get (pointer to) last directory in inpath (null if inpath does not exist)
  virtual MonitorElement * getDirectory(std::string inpath) const = 0;
  
  // look for object <name> in current directory
  //  virtual MonitorElement * findObject(std::string name) const = 0;
  // look for object <name> in directory <pathname>
  virtual MonitorElement * findObject(std::string name, 
				      std::string pathname) const = 0;
  // look for folder <name> in current directory
  virtual MonitorElement * findFolder(std::string name) const = 0;
    
  // ---------------- Miscellaneous -----------------------------
  
  // convert dqm::me_util::monit_map into 
  // vector<string> of the form: <dir pathname>:<obj1>,<obj2>,...
  // to be invoked by getAddedContents, getRemovedContents, getUpdatedContents
  void convert(std::vector<std::string> & put_here, 
	       const dqm::me_util::monit_map & in) const;
  
  /* come here at end of monitoring cycle for all receivers;
     (a) call resetUpdate for modified contents

     (b) if resetMEs=true, reset MEs that were updated (and have resetMe = true);
     [flag resetMe is typically set by sources (false by default)];
     [Clients in standalone mode should also have resetMEs = true] 

     (c) if callResetStuff = true, call resetStuff
     (typical behaviour: Sources & Collector have callResetStuff = true, whereas
     clients have callResetStuff = false, so GUI/WebInterface can access the 
     modifications in monitorable & monitoring) */
  void doneSending(bool resetMEs, bool callResetStuff);
 
  // ------------------- Booking ---------------------------

  // add monitoring element to directory <pathname>
  virtual void addElement(MonitorElement * me, std::string pathname, 
			  std::string type = "") = 0;
  // add null monitoring element to current folder (can NOT be folder);
  // used for registering monitorables before user has subscribed to <name>
  //virtual void addElement(std::string name) = 0;
  // add null monitoring element to folder <pathname> (can NOT be folder);
  // used for registering monitorables before user has subscribed to <name>
  virtual void addElement(std::string name, std::string pathname) = 0;

  // book 1D histogram
  virtual MonitorElement * book1D(std::string name, std::string title, 
			  int nchX, double lowX, double highX, 
			  MonitorElement * folder) = 0;

  // book 2D histogram
  virtual MonitorElement * book2D(std::string name, std::string title,
			  int nchX, double lowX, double highX, int nchY, 
			  double lowY, double highY, 
			  MonitorElement * folder) = 0;
  // book 3D histogram
  virtual MonitorElement * book3D(std::string name, std::string title,
			  int nchX, double lowX, double highX, int nchY, 
			  double lowY, double highY, int nchZ,
			  double lowZ, double highZ, 
			  MonitorElement * folder) = 0;
  // book profile;
  // option is one of: " ", "s" (default), "i", "G" (see TProfile::BuildOptions)
  // (in a profile plot the number of channels in Y is disregarded)
  virtual MonitorElement * bookProfile(std::string name, 
			       std::string title,int nchX, double lowX,
			       double highX, int nchY, double lowY, 
			       double highY, MonitorElement* folder,
				char * option = "s") = 0;

  // book 2-D profile;
  // option is one of: " ", "s" (default), "i", "G" (see TProfile2D::BuildOptions)
  // (in a 2-D profile plot the number of channels in Z is disregarded)
  virtual MonitorElement * bookProfile2D(std::string name, 
				 std::string title, 
				 int nchX, double lowX, double highX, 
				 int nchY, double lowY,double highY,
				 int nchZ, double lowZ,double highZ,
				 MonitorElement * folder, 
				 char * option = "s") = 0;

  // book float
  virtual MonitorElement * bookFloat(std::string s, MonitorElement * folder)=0;
  // book int
  virtual MonitorElement * bookInt(std::string s, MonitorElement * folder)=0;
  // book string
  virtual MonitorElement * bookString(std::string s, std::string v, 
			      MonitorElement * folder) = 0;


  // ---------------- Checkers -----------------------------
  
  // true if pathname exists
  virtual bool pathExists(std::string inpath) const = 0;
  // check against null objects (true if object exists)
  bool checkElement(const MonitorElement * const me) const;
  // check if object is really a folder (true if it is)
  bool checkFolder(const MonitorElement * const dir) const;
  // true if object <name> already belongs to directory <pathname>
  virtual bool objectDefined(std::string name, std::string pathname) const 
    = 0;
  
  // true if directory (or any subfolder at any level below it) contains
  // at least one valid (i.e. non-null) monitoring element
  virtual bool containsAnyMEs(std::string pathname) const = 0;
  // true if directory (or any subfolder at any level below it) contains
  // at least one monitorable element
  virtual bool containsAnyMonitorable(std::string pathname) const = 0;

  // true if Monitoring Element <me> is needed by any subscriber
  virtual bool isNeeded(std::string pathname, std::string me) const=0;

  // -------------------- Unsubscribing/Removing --------------------
  
  // remove monitoring element from directory;
  // if warning = true, print message if element does not exist
  virtual void removeElement(MonitorElement * dir, std::string name,  
			     bool warning = true) = 0;
  // remove all monitoring elements from directory;
  // if warning = true, print message if element does not exist
  virtual void removeContents(MonitorElement * dir, bool warning = true) = 0;
  
  // -------------------- Deleting ----------------------------------
  // delete directory and all contents;
  virtual void rmdir(MonitorElement * dir) = 0;
  
  // copy monitoring elements from source to destination
  virtual void copy(MonitorElement * const source, MonitorElement * const dest, 
		    const std::vector<std::string> & contents) = 0;
  // remove subscribed monitoring elements; 
  // if warning = true, printout error messages when problems;
  virtual void removeSubsc(MonitorElement * const dir, 
			   const std::vector<std::string> & contents, 
			   bool warning = true) = 0;
  
  // -------------------- Misc ----------------------------------

  // add <name> to back-end interface's updatedContents
  void add2UpdatedContents(std::string name, 
			   std::string pathname);

  // add (QReport) MonitorElement to back-end intereface's updatedQReports
  void add2UpdatedQReports(QReport * qr)
  {updatedQReports.insert(qr);}

  // (a) reset modifications to monitorable since last cycle 
  // (b) reset sets of added/removed/updated contents and updated QReports
  void resetStuff(void);

  pthread_mutex_t mutex_;

  // ------------------- data structures -----------------------------
  
  // directory structure of "this"
  dqm::me_util::MonitorStruct own;
  // directory structure of subscribers
  dqm::me_util::MonitorStruct subscribers;
  
  // holds (un)subscription requests that are not included in "own"; 
  // format: <dir pathname>:<obj1>,<obj2>,...
  // saved here by a downstream class, till ReceiverBase 
  // sends the request to the sender
  struct SubcRequests_ {
    LockMutex::Mutex mutex;
    std::list<std::string> toAdd; 
    std::list<std::string> toRemove; 
  };
  typedef struct SubcRequests_ SubcRequests;

  SubcRequests requests;

  // new added & removed monitorable since last cycle; 
  // format: <dir pathname>:<obj1>,<obj2>,...
  // reset after all recipients have been informed (ie. in resetStuff)
  std::vector<std::string> addedMonitorable;
  std::vector<std::string> removedMonitorable;
  // new added & removed contents since last cycle;
  // reset after all recipients have been informed (ie. in resetStuff);
  // Note: these do not include objects in subscriber's folders
  dqm::me_util::monit_map addedContents;
  dqm::me_util::monit_map removedContents;
  // updated monitoring elements since last cycle
  // format: <dir pathname>:<obj1>,<obj2>,...
  // *** Note: includes addedContents ***
  dqm::me_util::monit_map updatedContents;

  // map of all quality tests
  dqm::qtests::QC_map qtests_;

  // set of updated quality reports since last monitoring cycle
  std::set<QReport *> updatedQReports;

  // get "global" folder <inpath> status (one of: STATUS_OK, WARNING, ERROR, OTHER);
  // returns most sever error, where ERROR > WARNING > OTHER > STATUS_OK;
  // see Core/interface/QTestStatus.h for details on "OTHER" 
  virtual int getStatus(std::string inpath = "") const = 0;

  // ------------ Operations for MEs that are normally never reset ---------

  // reset contents (does not erase contents permanently)
  // (makes copy of current contents; will be subtracted from future contents)
  void softReset(MonitorElement * me);

  // reverts action of softReset
  void disableSoftReset(MonitorElement * me);

  // --- Operations on MEs that are normally reset at end of monitoring cycle ---

  // if true, will accumulate ME contents (over many periods)
  // until method is called with flag = false again
  void setAccumulate(MonitorElement * me, bool flag);

  // universal verbose flag for DQM
  unsigned DQM_VERBOSE;

  // -------------------- Quality tests on MonitorElements ------------------

   // add quality report (to be called when test is to run locally)
  virtual QReport * addQReport(MonitorElement * me, QCriterion * qc) const = 0;
  // add quality report (to be called by ReceiverBase)
  virtual QReport * addQReport(MonitorElement * me, std::string qtname,
			       QCriterion * qc = 0) const = 0;

  // add quality report to ME
  void addQReport(MonitorElement * me, QReport * qr) const
  {me->addQReport(qr);}

  // check if QReport is already defined for ME
  bool qreportExists(MonitorElement * me, std::string qtname) const
  {return me->qreportExists(qtname);}

  // get QCriterion corresponding to <qtname> 
  // (null pointer if QCriterion does not exist)
  QCriterion * getQCriterion(std::string qtname) const;

  // get QReport from ME (null pointer if no such QReport)
  QReport * getQReport(MonitorElement * me, std::string qtname);

  // run quality tests (also finds updated contents in last monitoring cycle,
  // including newly added content) 
  void runQTests(void);

  // create quality test with unique name <qtname> (analogous to ME name);
  // quality test can then be attached to ME with useQTest method
  // (<algo_name> must match one of known algorithms)
  virtual QCriterion * createQTest(std::string algo_name,
				   std::string qtname) = 0;

  // attach quality test <qtname> to all ME matching <search_string>;
  // <search_string> could : (a) be exact pathname (e.g. A/B/C/histo)
  // (b) include wildcards (e.g. A/?/C/histo, A/B/*/histo or A/B/*);
  // this action applies to all MEs already available or future ones
  void useQTest(std::string search_string, std::string qtname) const;

  // look for all MEs matching <search_string> in own.global_;
  // if found, create QReport from QCriterion and add to ME
  void scanContents(QCriterion * qc, const std::string & search_string) const;

  // same as scanContents above but for one path only
  void scanContents(QCriterion * qc, const std::string & search_string,
		    dqm::me_util::cglob_it & path) const;

  // check if resetStuff was called 
  // (to be reset in MonitorUserInterface::runQualityTests)
  inline bool wasResetCalled() const{return resetWasCalled;}

 private:
  // use to printout warning when calling quality tests twice without
  // having called resetStuff in between...
  // (to be reset in MonitorUserInterface::runQualityTests)
  bool resetWasCalled;
  // run quality tests (also finds updated contents in last monitoring cycle,
  // including newly added content) <-- to be called only by runQTests
  void runQualityTests(void);

  // loop over quality tests & addedContents: look for MEs that 
  // match QCriterion::searchStrings; upon a match, add QReport to ME(s)
  void checkAddedElements(void);
 
  // loop over addedContents: look for MEs that 
  // match search_string; upon a match, add QReport to ME(s)
  void checkAddedElements(const std::string & search_string, 
			  dqm::qtests::cqc_it & qc);

  // same as checkAddedElements above for only one path
  void checkAddedElements(const std::string & search_string, 
			  dqm::qtests::cqc_it & qc,
			  dqm::me_util::cmonit_it & path);
 
  DaqMonitorBEInterface(const DaqMonitorBEInterface&);
  const DaqMonitorBEInterface& operator=(const DaqMonitorBEInterface&);

  friend class NodeBase;
  friend class SenderBase;
  friend class ReceiverBase;
  friend class MonitorUserInterface;
  friend class MonitorUIRoot;

  // this is really bad; unfortunately, gcc 3.2.3 won't let me define 
  // template classes, so I have to find a workaround for now
  // error: "...is not a template type" - christos May26, 2005
  friend class CollateMET;
  friend class CollateMERootH1;
  friend class CollateMERootH2;
  friend class CollateMERootH3;
  friend class CollateMERootProf;
  friend class CollateMERootProf2D;

};


#endif
