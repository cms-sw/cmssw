#ifndef DaqMonitorBEInterface_h
#define DaqMonitorBEInterface_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/GlobalMutex.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/StringUtil.h"
#include "DQMServices/Core/interface/QCriterion.h"
#include "DQMServices/Core/interface/QTestStatus.h"

#include <iostream>
#include <vector>
#include <list>
#include <map>
#include <string>

class QCriterion;
class MonitorElementRootFolder;
class DQMTagHelper;
class CollateMonitorElement;
class TObject;
class TH1F;
class TH2F;
class TH3F;
class TProfile;
class TProfile2D;

namespace edm {
  class DQMHttpSource;
}

class DaqMonitorBEInterface: public StringUtil
{

 public:
  
  DaqMonitorBEInterface(edm::ParameterSet const &pset)
  {
    //pthread_mutex_init(&mutex_,0); 
    dqm_locker = 0;
    DQM_VERBOSE = 1; resetMonitoringDiff(); resetMonitorableDiff();
  }  
  virtual ~DaqMonitorBEInterface();
 
  void reParseConfig(const edm::ParameterSet &pset){}
 
 
  /// book 1D histogram
  virtual MonitorElement * book1D(std::string name, 
				  std::string title, 
				  int nchX, double lowX, double highX)=0;
  /// book 1D variable bin histogram
  virtual MonitorElement * book1D(std::string name,
				  std::string title,
				  int nchX, float *xbinsize)=0;
  /// book 2D histogram
  virtual MonitorElement * book2D(std::string name, 
				  std::string title, 
				  int nchX, double lowX, double highX, 
				  int nchY, double lowY, double highY)=0;
  /// book 3D histogram
  virtual MonitorElement * book3D(std::string name, 
				  std::string title, 
				  int nchX, double lowX, double highX, 
				  int nchY, double lowY, double highY, 
				  int nchZ, double lowZ, double highZ)=0;
  /// book profile; 
  /// option is one of: " ", "s" (default), "i", "G" (see TProfile::BuildOptions)
  /// (in a profile plot the number of channels in Y is disregarded)
  virtual MonitorElement * bookProfile(std::string name, 
				       std::string title, 
				       int nchX, double lowX, double highX, 
				       int nchY, double lowY,double highY,
				       char * option = "s")=0;
  /// book 2-D profile;
  /// option is one of: " ", "s" (default), "i", "G" (see TProfile2D::BuildOptions)
  /// (in a 2-D profile plot the number of channels in Z is disregarded)
  virtual MonitorElement * bookProfile2D(std::string name, 
					 std::string title, 
					 int nchX, double lowX, double highX,
					 int nchY, double lowY,double highY,
					 int nchZ, double lowZ,double highZ,
					 char * option = "s") = 0;
  /// book float
  virtual MonitorElement * bookFloat(std::string ) = 0;
  /// book int
  virtual MonitorElement * bookInt(std::string ) = 0;
  /// book string
  virtual MonitorElement * bookString(std::string, std::string )=0;
  
  // ---------------- Navigation -----------------------
  
  /// retun pathname of current directory
  virtual std::string pwd(void) const = 0;
  /// go to top directory (ie. root)
  virtual void cd(void) = 0;
  /// equivalent to "cd .."
  virtual void goUp(void) = 0;
  /// set the last directory in fullpath as the current directory (create if needed)
  /// to be invoked by user to specify directories for monitoring objects b4 booking
  /// commands book1D (etc) & removeElement(name) imply elements in this directory!
  virtual void setCurrentFolder(std::string fullpath)=0;
  /// cd to subdirectory (if there)
  virtual void cd(std::string subdir_path) = 0;
  /// name of global monitoring folder (containing all sources subdirectories)
  static const std::string monitorDirName;
  static const std::string referenceDirName;
  static const std::string collateDirName;
  static const std::string dqmPatchVersion;
  // ---------------- Miscellaneous -----------------------------
  
  /// true if directory exists
  bool dirExists(std::string inpath) const
  {return pathExists(inpath, Own);}
  /// show directory structure
  virtual void showDirStructure(void) const = 0;
  /// save directory with monitoring objects into root file <filename>;
  /// include quality test results with status >= minimum_status 
  /// (defined in Core/interface/QTestStatus.h);
  /// if directory="", save full monitoring structure
  virtual void save(std::string filename, std::string directory="",
		    int minimum_status=dqm::qstatus::STATUS_OK) = 0;
  /// open/read root file <filename>, and copy MonitorElements;
  /// if flag=true, overwrite identical MonitorElements (default: false);
  /// if directory != "", read only selected directory
  /// if prepend !="", prepend string to the path
  virtual void open(std::string filename, bool overwrite = false,
		    std::string directory="", std::string prepend="") = 0;

  // ------------------- Reference ME -------------------------------
  
  virtual void readReferenceME(std::string filename) = 0 ;
  virtual bool makeReferenceME(MonitorElement* me) = 0 ;
  virtual bool isCollateME(MonitorElement* me) const = 0 ;
  virtual bool isReferenceME(MonitorElement* me) const = 0 ;
  virtual MonitorElement* getReferenceME(MonitorElement* me) const = 0 ;
  virtual void deleteME(MonitorElement* me) = 0 ;

  // ------------------- File Versioning ---------------------------

  virtual std::string getFileReleaseVersion(std::string filename) = 0 ;
  virtual std::string getFileDQMPatchVersion(std::string filename) = 0 ;
  std::string getDQMPatchVersion() { return "DQMPATCH:"+dqmPatchVersion; } 

  // ---------------------------------------------------------------
  
  /// cycle through all monitoring objects, draw one at time
  virtual void drawAll(void) = 0;
  
  /// get list of subdirectories of current directory
  virtual std::vector<std::string> getSubdirs(void) const = 0;
  /// get list of (non-dir) MEs of current directory
  virtual std::vector<std::string> getMEs(void) const = 0;
  /// set verbose level (0 turns all non-error messages off)
  void setVerbose(unsigned level){DQM_VERBOSE = level;}
  /// get verbose level
  unsigned getVerbose(void) const {return DQM_VERBOSE;}

  // -------------------- Deleting ----------------------------------
  
  /// remove directory
  virtual void rmdir(std::string fullpath) = 0;
  /// erase monitoring element in current directory 
  /// (opposite of book1D,2D,etc. action);
  virtual void removeElement(std::string name) = 0;
  /// erase all monitoring elements in current directory (not including subfolders);
  virtual void removeContents(void) = 0;
  
  /// acquire and release lock
  void lock();
  void unlock();

  // ------------- Tags (e.g. detector-IDs;) -----------------
  // Property similar to "Labels" of google-mail) 

  /// tag ME as <myTag> (myTag > 0)
  virtual void tag(MonitorElement * me, unsigned int myTag) = 0;
  /// opposite action of tag method (myTag > 0)
  virtual void untag(MonitorElement * me, unsigned int myTag) = 0;
  /// tag ME specified by full pathname (e.g. "my/long/dir/my_histo")
  virtual void tag(std::string fullpathname, unsigned int myTag)= 0;
  /// opposite action of tag method
  virtual void untag(std::string fullpathname, unsigned int myTag)= 0;
  /// tag all children of folder (does NOT include subfolders)
  virtual void tagContents(std::string pathname, unsigned int myTag)= 0;
  /// opposite action of tagContents method
  virtual void untagContents(std::string pathname, unsigned int myTag)=0;
  /// tag all children of folder, including all subfolders and their children;
  /// exact pathname: FAST
  /// pathname including wildcards (*, ?): SLOW!
  virtual void tagAllContents(std::string pathname, unsigned int myTag)=0;
  /// opposite action of tagAllContents method
  virtual void untagAllContents(std::string pathname, unsigned int myTag)=0;

  // ------------------- Public "getters" ------------------------------
  /// get ME from full pathname (e.g. "my/long/dir/my_histo")
  virtual MonitorElement * get(std::string fullpath) const = 0;

  /// get all MonitorElements tagged as <tag>
  virtual std::vector<MonitorElement *> get(unsigned int tag) const = 0;

  /// get vector with all children of folder
  /// (does NOT include contents of subfolders)
  virtual std::vector<MonitorElement *> getContents(std::string pathname) 
    const = 0;
  /// same as above for tagged MonitorElements
  virtual std::vector<MonitorElement *> getContents
    (std::string pathname, unsigned int tag) const=0;

  /// get vector with children of folder, including all subfolders + their children;
  /// exact pathname: FAST
  /// pathname including wildcards (*, ?): SLOW!
  virtual std::vector<MonitorElement*> getAllContents(std::string pathname)
    const = 0;
  /// same as above for tagged MonitorElements
  virtual std::vector<MonitorElement*> getAllContents(std::string pathname,
						      unsigned int tag) 
  const = 0;
  

/// un-protected to enable full use of this service class, A.Meyer 070814
// protected:
  
  // ------------------- Private "getters" ------------------------------
  
  /// add all (tagged) MEs to put_here
  virtual void get(const dqm::me_util::dir_map & Dir, 
		   std::vector<MonitorElement *> & put_here) const=0;

  /// get all contents;
  /// return vector<string> of the form <dir pathname>:<obj1>,<obj2>,<obj3>;
  /// if showContents = false, change form to <dir pathname>:
  /// (useful for subscription requests; meant to imply "all contents")
  virtual void getContents(std::vector<std::string> & put_here,
			   bool showContents = true) const = 0;
  /// get monitorable;
  /// return vector<string> of the form <dir pathname>:<obj1>,<obj2>,<obj3>;
  /// if showContents = false, change form to <dir pathname>:
  /// (useful for subscription requests; meant to imply "all contents")
  virtual void getMonitorable(std::vector<std::string> & put_here,
			      bool showContents = true) const = 0;
  /// get added monitorable (since last cycle)
  /// return vector<string> of the form: <dir pathname>:<obj1>,<obj2>,...
  void getAddedMonitorable(std::vector<std::string> & put_here) const;
  /// get removed monitorable (since last cycle)
  /// return vector<string> of the form: <dir pathname>:<obj1>,<obj2>,...
  void getRemovedMonitorable(std::vector<std::string> & put_here) const;
  /// get added contents (since last cycle)
  /// return vector<string> of the form: <dir pathname>:<obj1>,<obj2>,...
  void getAddedContents(std::vector<std::string> & put_here) const;
  /// get removed contents (since last cycle)
  /// return vector<string> of the form: <dir pathname>:<obj1>,<obj2>,...
  void getRemovedContents(std::vector<std::string> & put_here) const;
  /// get updated contents (since last cycle)
  /// COMPLEMENTARY to addedContents, removedContents
  /// return vector<string> of the form: <dir pathname>:<obj1>,<obj2>,...
  void getUpdatedContents(std::vector<std::string> & put_here) const;
 
  /// get folder corresponding to inpath wrt to root (create subdirs if necessary)
  virtual MonitorElementRootFolder * 
    makeDirectory(std::string inpath, dqm::me_util::rootDir & Dir) = 0;
  /// get (pointer to) last directory in inpath (null if inpath does not exist)
  virtual MonitorElementRootFolder * 
    getDirectory(std::string inpath, const dqm::me_util::rootDir & Dir) 
    const = 0;
  /// get folder corresponding to inpath wrt to root (create subdirs if necessary)
  MonitorElementRootFolder * makeDirectory(std::string inpath)
  {return makeDirectory(inpath, Own);}
  /// get (pointer to) last directory in inpath (null if inpath does not exist)
  MonitorElementRootFolder * getDirectory(std::string inpath) const
  {return getDirectory(inpath, Own);}
  
  /// look for object <name> in current directory
  ///  virtual MonitorElement * findObject(std::string name) const = 0;
  /// look for object <name> in directory <pathname>
  virtual MonitorElement * findObject(std::string name, 
				      std::string pathname) const = 0;
  /// look for folder <name> in current directory
  virtual MonitorElement * findFolder(std::string name) const = 0;
    
  /// get root folder
  virtual MonitorElementRootFolder * getRootFolder
    (const dqm::me_util::rootDir & Dir) const = 0;

  /** get tags for various maps, return vector with strings of the form
      <dir pathname>:<obj1>/<tag1>/<tag2>,<obj2>/<tag1>/<tag3>, etc. */
  virtual void getAllTags(std::vector<std::string> & put_here) const = 0;
  virtual void getAddedTags(std::vector<std::string> & put_here) const = 0;
  virtual void getRemovedTags(std::vector<std::string> & put_here) const = 0;

  /// get vector with all children of folder in <rDir>
  /// (does NOT include contents of subfolders)
  virtual void getContents
    (std::string & pathname, const dqm::me_util::rootDir & rDir,
     std::vector<MonitorElement *> & put_here) const = 0;

  /// get vector with all children of folder and all subfolders of <rDir>;
  /// pathname may include wildcards (*, ?) ==> SLOW!
  virtual void getAllContents
    (std::string & pathname, const dqm::me_util::rootDir & rDir,
     std::vector<MonitorElement*> & put_here) const = 0;

  /// get rootDir corresponding to tag 
  /// (Own for tag=0, or null for non-existing tag)
  const dqm::me_util::rootDir * getRootDir(unsigned int tag) const;

  /// check if added contents match rules; put matches in put_here
  void checkAddedContents(const dqm::me_util::searchCriteria & rules,
			  std::vector<MonitorElement *> & put_here) const;
  /// same as above for given search_string and path; 
  /// put matches into put_here
  virtual void checkAddedContents
    (const std::string & search_string, 
     dqm::me_util::cmonit_it& added_path,const dqm::me_util::rootDir & Dir,
     std::vector<MonitorElement*> & put_here) const = 0;
  /// check if added contents match search paths
  virtual void checkAddedSearchPaths
    (const std::vector<std::string> & search_path, 
     const dqm::me_util::rootDir & Dir,
     std::vector<MonitorElement*>& put_here)const=0;
  /// check if added contents belong to folders 
  /// (use flag to specify if subfolders should be included)
  void checkAddedFolders(const std::vector<std::string> & folders,
			 const dqm::me_util::rootDir & Dir,
			 bool useSubfolders,
			 std::vector<MonitorElement*>& put_here) const;
  /// check if added contents belong to folder 
  /// (use flag to specify if subfolders should be included)
  virtual void checkAddedFolder
    (dqm::me_util::cmonit_it & added_path,
     const dqm::me_util::rootDir & Dir,
     std::vector<MonitorElement*>& put_here)const=0;
  /// check if added contents are tagged
  void checkAddedTags(const dqm::me_util::rootDir & Dir,
		      std::vector<MonitorElement*>& put_here) const;

  // ---------------- Miscellaneous -----------------------------
  
  /// convert dqm::me_util::monit_map into 
  /// vector<string> of the form: <dir pathname>:<obj1>,<obj2>,...
  /// to be invoked by getAddedContents, getRemovedContents, getUpdatedContents
  void convert(std::vector<std::string> & put_here, 
	       const dqm::me_util::monit_map & in) const;
  
  /** come here after sending monitoring to all receivers;
     (a) call resetUpdate for modified contents:

     if resetMEs=true, reset MEs that were updated (and have resetMe = true);
     [flag resetMe is typically set by sources (false by default)];
     [Clients in standalone mode should also have resetMEs = true] 

     (b) if callResetDiff = true, call resetMonitoringDiff
     (typical behaviour: Sources & Collector have callResetDiff = true, whereas
     clients have callResetDiff = false, so GUI/WebInterface can access the 
     modifications in monitorable & monitoring) */
  void doneSendingMonitoring(bool resetMEs, bool callResetDiff);
  /** come here after sending monitorable to all receivers;
     if callResetDiff = true, call resetMonitorableDiff
     (typical behaviour: Sources & Collector have callResetDiff = true, whereas
     clients have callResetDiff = false, so GUI/WebInterface can access the 
     modifications in monitorable & monitoring) */
  void doneSendingMonitorable(bool callResetDiff);
  /// extract object (TH1F, TH2F, ...) from <to>; return success flag
  /// flag fromRemoteNode indicating if ME arrived from different node
  virtual bool extractObject(TObject * to, MonitorElementRootFolder * dir, 
			     bool fromRemoteNode)=0;
  /// true if Monitoring Element <me> in directory <folder> has isDesired = true;
  /// if warning = true and <me> does not exist, show warning
  virtual bool isDesired(MonitorElementRootFolder * folder, 
		 std::string me, bool warning) const=0;
 
  // ------------------- Booking ---------------------------

  /// add monitoring element to directory <pathname>
  virtual void addElement(MonitorElement * me, std::string pathname, 
			  std::string type = "") = 0;
  /// add null monitoring element to folder <pathname> (can NOT be folder);
  /// used for registering monitorables before user has subscribed to <name>
  virtual void addElement(std::string name, std::string pathname) = 0;

  /// book 1D histogram
  virtual MonitorElement * book1D(std::string name, std::string title, 
			  int nchX, double lowX, double highX, 
			  MonitorElementRootFolder * folder) = 0;

  /// book 1D variable bin histogram
  virtual MonitorElement * book1D(std::string name, std::string title,
			  int nchX, float *xbinsize,
			  MonitorElementRootFolder * folder) = 0;

  /// book 2D histogram
  virtual MonitorElement * book2D(std::string name, std::string title,
			  int nchX, double lowX, double highX, int nchY, 
			  double lowY, double highY, 
			  MonitorElementRootFolder * folder) = 0;
  /// book 3D histogram
  virtual MonitorElement * book3D(std::string name, std::string title,
			  int nchX, double lowX, double highX, int nchY, 
			  double lowY, double highY, int nchZ,
			  double lowZ, double highZ, 
			  MonitorElementRootFolder * folder) = 0;
  /// book profile;
  /// option is one of: " ", "s" (default), "i", "G" (see TProfile::BuildOptions)
  /// (in a profile plot the number of channels in Y is disregarded)
  virtual MonitorElement * bookProfile(std::string name, 
			       std::string title,int nchX, double lowX,
			       double highX, int nchY, double lowY, 
			       double highY, MonitorElementRootFolder* folder,
				char * option = "s") = 0;

  /// book 2-D profile;
  /// option is one of: " ", "s" (default), "i", "G" (see TProfile2D::BuildOptions)
  /// (in a 2-D profile plot the number of channels in Z is disregarded)
  virtual MonitorElement * bookProfile2D(std::string name, 
				 std::string title, 
				 int nchX, double lowX, double highX, 
				 int nchY, double lowY,double highY,
				 int nchZ, double lowZ,double highZ,
				 MonitorElementRootFolder * folder, 
				 char * option = "s") = 0;

  /// book float
  virtual MonitorElement * bookFloat(std::string s, 
				     MonitorElementRootFolder * folder)=0;
  /// book int
  virtual MonitorElement * bookInt(std::string s, 
				   MonitorElementRootFolder * folder)=0;
  /// book string
  virtual MonitorElement * bookString(std::string s, std::string v, 
				      MonitorElementRootFolder * folder) = 0;
  

  // ---------------- Checkers -----------------------------
  
  /// true if pathname exists
  virtual bool pathExists(std::string inpath, 
			  const dqm::me_util::rootDir & Dir) const = 0;
  /// check against null objects (true if object exists)
  bool checkElement(const MonitorElement * const me) const;
  
  /// true if directory (or any subfolder at any level below it) contains
  /// at least one valid (i.e. non-null) monitoring element
  virtual bool containsAnyMEs(std::string pathname) const = 0;
  /// true if directory (or any subfolder at any level below it) contains
  /// at least one monitorable element
  virtual bool containsAnyMonitorable(std::string pathname) const = 0;

  /// true if Monitoring Element <me> is needed by any subscriber
  virtual bool isNeeded(std::string pathname, std::string me) const=0;

  // -------------------- Unsubscribing/Removing --------------------
  
  /// remove monitoring element from directory;
  /// if warning = true, print message if element does not exist
  virtual void removeElement(MonitorElementRootFolder * dir, 
			     std::string name,  
			     bool warning = true) = 0;
  /// remove all monitoring elements from directory;
  /// if warning = true, print message if element does not exist
  virtual void removeContents(MonitorElementRootFolder * dir) = 0;
  
  /// remove all contents from <pathname> from all subscribers, tags and CMEs
  void removeCopies(const std::string & pathname);
  /// remove Monitor Element <name> from all subscribers, tags & CME directories
  void removeCopies(const std::string & pathname, const std::string & name);
  /// remove Monitor Element <name> from <pathname> in <Dir>
  void remove(const std::string & pathname, const std::string & name,
		  dqm::me_util::rootDir & Dir);
  // -------------------- Deleting ----------------------------------
  /// delete directory and all contents;
  /// delete directory (all contents + subfolders);
  virtual void rmdir
    (const std::string & pathname, dqm::me_util::rootDir & rDir) = 0;
 
  /// copy monitoring elements from source to destination
  virtual void copy(const MonitorElementRootFolder * const source, 
		    MonitorElementRootFolder * const dest, 
		    const std::vector<std::string> & contents) = 0;
  /// remove subscribed monitoring elements; 
  /// if warning = true, printout error messages when problems;
  virtual void removeSubsc(MonitorElementRootFolder * const dir, 
			   const std::vector<std::string> & contents, 
			   bool warning = true) = 0;
  
  // -------------------- Misc ----------------------------------

  /// add <name> to back-end interface's updatedContents
  void add2UpdatedContents(std::string name, 
			   std::string pathname);

  /// add (QReport) MonitorElement to back-end intereface's updatedQReports
  void add2UpdatedQReports(QReport * qr)
  {updatedQReports.insert(qr);}

  /// reset modifications to monitorable since last cycle 
  /// and sets of added/removed contents
  void resetMonitorableDiff();

  /// reset updated contents and updated QReports
  void resetMonitoringDiff();
      
  boost::mutex::scoped_lock * dqm_locker;
  // ------------------- data structures -----------------------------
  
  /// directory monitoring structure for all MEs
  dqm::me_util::rootDir Own;

  /// directory structure of subscribers
  dqm::me_util::subscriber_map Subscribers;

  /// directory structure of tags (eg. detector-IDs)
  dqm::me_util::tag_map Tags;

  /// holds (un)subscription requests that are not included in "own"; 
  /// format: <dir pathname>:<obj1>,<obj2>,...
  /// saved here by a downstream class, till ReceiverBase 
  /// sends the request to the sender
  struct SubcRequests_ {
    LockMutex::Mutex mutex;
    std::list<std::string> toAdd; 
    std::list<std::string> toRemove; 
  };
  typedef struct SubcRequests_ SubcRequests;

  SubcRequests requests;

  /// new added & removed monitorable since last cycle; 
  /// format: <dir pathname>:<obj1>,<obj2>,...
  /// reset after all recipients have been informed (ie. in resetStuff)
  std::vector<std::string> addedMonitorable;
  std::vector<std::string> removedMonitorable;
  /// new added & removed contents since last cycle;
  /// reset after all recipients have been informed (ie. in resetStuff);
  /// Note: these do not include objects in subscriber's folders
  dqm::me_util::monit_map addedContents;
  dqm::me_util::monit_map removedContents;
  /// all tags for full monitoring structure 
  dqm::me_util::dir_tags allTags;
  /// added and removed tags since last cycle;
  /// reset after all recipients have been informed (ie. in resetStuff);
  dqm::me_util::dir_tags addedTags;
  dqm::me_util::dir_tags removedTags;

  /// updated monitoring elements since last cycle
  /// format: <dir pathname>:<obj1>,<obj2>,...
  /// *** Note: includes addedContents ***
  dqm::me_util::monit_map updatedContents;


  /// map of all quality tests
  dqm::qtests::QC_map qtests_;

  /// set of updated quality reports since last monitoring cycle
  std::set<QReport *> updatedQReports;

  /// get "global" folder <inpath> status (one of:STATUS_OK, WARNING, ERROR, OTHER);
  /// returns most sever error, where ERROR > WARNING > OTHER > STATUS_OK;
  /// see Core/interface/QTestStatus.h for details on "OTHER" 
  virtual int getStatus(std::string inpath = "") const = 0;
  /// same as above for tag;
  virtual int getStatus(unsigned int tag) const = 0;
  /// same as above for vector with MonitorElements
  int getStatus(const std::vector<MonitorElement *> & ME_group) const;

  /// true if at least one ME gave hasError/hasWarning/hasOtherReport = true
  bool hasError(const std::vector<MonitorElement *> & ME_group) const;
  bool hasWarning(const std::vector<MonitorElement *> & ME_group) const;
  bool hasOtherReport(const std::vector<MonitorElement *> & ME_group) const;
  
  // ------------ Operations for MEs that are normally never reset ---------

  /// reset contents (does not erase contents permanently)
  /// (makes copy of current contents; will be subtracted from future contents)
  void softReset(MonitorElement * me);

  /// reverts action of softReset
  void disableSoftReset(MonitorElement * me);

  // --- Operations on MEs that are normally reset at end of monitoring cycle ---

  /// if true, will accumulate ME contents (over many periods)
  /// until method is called with flag = false again
  void setAccumulate(MonitorElement * me, bool flag);

  /// universal verbose flag for DQM
  unsigned DQM_VERBOSE;

  // -------------------- Quality tests on MonitorElements ------------------

   /// add quality report (to be called when test is to run locally)
  virtual QReport * addQReport(MonitorElement * me, QCriterion * qc) const = 0;
  /// add quality report (to be called by ReceiverBase)
  virtual QReport * addQReport(MonitorElement * me, std::string qtname,
			       QCriterion * qc = 0) const = 0;

  /// add quality report to ME
  void addQReport(MonitorElement * me, QReport * qr) const
  {me->addQReport(qr);}

  /// add quality reports to all MEs
  void addQReport
    (std::vector<MonitorElement *> & allMEs, QCriterion * qc) const;

  /// check if QReport is already defined for ME
  bool qreportExists(MonitorElement * me, std::string qtname) const
  {return me->qreportExists(qtname);}

  /// get QCriterion corresponding to <qtname> 
  /// (null pointer if QCriterion does not exist)
  QCriterion * getQCriterion(std::string qtname) const;

  /// get QReport from ME (null pointer if no such QReport)
  QReport * getQReport(MonitorElement * me, std::string qtname);

  /// run quality tests (also finds updated contents in last monitoring cycle,
  /// including newly added content) 
  void runQTests(void);

  /// create quality test with unique name <qtname> (analogous to ME name);
  /// quality test can then be attached to ME with useQTest method
  /// (<algo_name> must match one of known algorithms)
  virtual QCriterion * createQTest(std::string algo_name,
				   std::string qtname) = 0;

  /// attach quality test <qc> to all ME matching <search_string>;
  /// if tag != 0, this applies to tagged contents
  /// <search_string> could : (a) be exact pathname (e.g. A/B/C/histo): FAST
  /// (b) include wildcards (e.g. A/?/C/histo, A/B/*/histo or A/B/*): SLOW
  void useQTest(unsigned int tag, std::string search_string, 
		const dqm::me_util::rootDir & Dir, QCriterion * qc) const;

  /// attach quality test <qc> to directory contents ==> FAST
  /// if tag != 0, this applies to tagged contents
  /// (need exact pathname without wildcards, e.g. A/B/C);
  ///
  void useQTest(std::string search_string, std::string qtname) const;
  ///
  void useQTest(unsigned int tag, std::string search_string, 
              std::string qtname) const;
  ///
  void useQTest(unsigned int tag, std::string qtname) const;
    
  /// use flag to specify whether subfolders (& their contents) should be included;
  void useQTest(unsigned int tag, std::string pathname, bool useSubfolds, 
		const dqm::me_util::rootDir & Dir, QCriterion * qc) const;

  /// attach quality test <qtname> to tagged MEs ==> FAST
  void useQTest(unsigned int tag, const dqm::me_util::rootDir & Dir,
		QCriterion * qc) const;

  /// scan structure <rDir>, looking for all MEs matching <search_string>;
  /// put results in <put_here>
  void scanContents(const std::string & search_string, 
		    const dqm::me_util::rootDir & rDir,
		    std::vector<MonitorElement *> & put_here) const;
  /// same as scanContents above but for one path only
  virtual void scanContents(const std::string & search_string, 
			    const MonitorElementRootFolder * folder,
			    std::vector<MonitorElement *> & put_here) 
    const=0;

  /// look for all MEs matching <search_string> in Own;
  /// if found, create QReport from QCriterion and add to ME
  void scanContents(QCriterion * qc, const std::string & search_string) const;

  /// check if resetMonitoringDiff and resetMonitorableDiff were called 
  /// (to be reset in MonitorUserInterface::runQualityTests)
  inline bool wasResetCalled() const
  {return rMonitoringDiffWasCalled && rMonitorableDiffWasCalled;}

  /// make new directory structure for Subscribers, Tags and CMEs
  virtual void makeDirStructure
    (dqm::me_util::rootDir & Dir, std::string name)=0;

  DQMTagHelper * tagHelper;
  /// map of collation MEs (used to see if a ME is really a CME)
  dqm::me_util::cme_map collate_map;
  /// set of collation MEs
  dqm::me_util::cme_set collate_set;
  /// remove all CMEs
  void removeCollates();
  /// remove CME
  void removeCollate(CollateMonitorElement * cme);
  //
 private:
 
 
   // ---------------------- Booking ------------------------------------
  //book 1D histogram using existing histogram
  virtual MonitorElement * clone1D(std::string name, TH1F* source) = 0 ;
  //book 2D histogram using existing histogram
  virtual MonitorElement * clone2D(std::string name, TH2F* source) = 0 ;

  //book 3D histogram using existing histogram
  virtual MonitorElement * clone3D(std::string name, TH3F* source) = 0 ;

  //book TProfile using existing profile
  virtual MonitorElement * cloneProfile(std::string name, TProfile* source) = 0;

  //book TProfile2D using existing profile
  virtual MonitorElement * cloneProfile2D(std::string name, TProfile2D* source)
  = 0 ;

  /// use to printout warning when calling quality tests twice without
  /// having called resetMonitoringDiff, resetMonitorableDiff in between...
  /// (to be reset in MonitorUserInterface::runQualityTests)
  bool rMonitoringDiffWasCalled;
  bool rMonitorableDiffWasCalled;
  /// run quality tests (also finds updated contents in last monitoring cycle,
  /// including newly added content) <-- to be called only by runQTests
  virtual void runQualityTests(void) = 0;

  /// loop over quality tests & addedContents: look for MEs that 
  /// match QCriterion::rules; upon a match, add QReport to ME(s)
  void checkAddedElements(void);
  //
  DaqMonitorBEInterface(const DaqMonitorBEInterface&);
  const DaqMonitorBEInterface& operator=(const DaqMonitorBEInterface&);

  friend class NodeBase;
  friend class SenderBase;
  friend class ReceiverBase;
  friend class MonitorUserInterface;
  friend class MonitorUIRoot;
  friend class DQMTagHelper;

  // this is really bad; unfortunately, gcc 3.2.3 won't let me define 
  // template classes, so I have to find a workaround for now
  // error: "...is not a template type" - christos May26, 2005
  friend class CollateMonitorElement;
  friend class CollateMET;
  friend class CollateMERootH1;
  friend class CollateMERootH2;
  friend class CollateMERootH3;
  friend class CollateMERootProf;
  friend class CollateMERootProf2D;

  friend class edm::DQMHttpSource;
  
  friend class EDMtoMEConverter;
  friend class MEtoEDMConverter;
};


#endif
