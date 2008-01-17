#ifndef DaqMonitorBEInterface_h
#define DaqMonitorBEInterface_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/GlobalMutex.h"

#include "DQMServices/Core/interface/DQMDefinitions.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/MonitorElementRootT.h"

#include "DQMServices/Core/interface/StringUtil.h"
#include "DQMServices/Core/interface/QCriterion.h"
#include "DQMServices/Core/interface/QTestStatus.h"

#include <iostream>
#include <vector>
#include <list>
#include <map>
#include <string>
#include <set>

#include <TH1F.h>
#include <TH2F.h>
#include <TH3F.h>
#include <TProfile.h>
#include <TProfile2D.h>

#include <TFile.h>
#include <TObject.h>
#include <TClass.h>

namespace edm {
  class DQMHttpSource;
}

class DaqMonitorBEInterface: public StringUtil
{

 public:
  
//-------------------------------------------------------------------------
// ---------------------- Constructors ------------------------------------
  DaqMonitorBEInterface(edm::ParameterSet const &pset);
  virtual ~DaqMonitorBEInterface();

//-------------------------------------------------------------------------
  /// set verbose level (0 turns all non-error messages off)
  void setVerbose(unsigned level){DQM_VERBOSE = level;}

// ---------------------- public navigation -------------------------------
  /// set the last directory in fullpath as the current directory(create if needed);
  /// to be invoked by user to specify directories for monitoring objects 
  /// before booking;
  /// commands book1D (etc) & removeElement(name) imply elements in this directory!;
  void setCurrentFolder(std::string fullpath);
  /// cd to subdirectory (if there)
  void cd(std::string subdir_path);
  /// go to top directory (ie. root)
  void cd(void) {setCurrentFolder("/");}
  /// return pathname of current directory
  std::string pwd(void) const {return fCurrentFolder->getPathname();}
  /// equivalent to "cd .."
  void goUp(void);

  /// get list of subdirectories of current directory
  std::vector<std::string> getSubdirs(void) const;
  /// get list of (non-dir) MEs of current directory
  std::vector<std::string> getMEs(void) const;

  /// true if directory (or any subfolder at any level below it) contains
  /// at least one valid (i.e. non-null) monitoring element
  bool containsAnyMEs(std::string pathname) const;
  /// true if directory (or any subfolder at any level below it) contains
  /// at least one monitorable element
  bool containsAnyMonitorable(std::string pathname) const;
  /// true if directory exists
  bool dirExists(std::string inpath) const {return pathExists(inpath, Own);}
//-------------------------------------------------------------------------
// ---------------------- public ME booking -------------------------------

  /// book 1D histogram
  MonitorElement * book1D(std::string name, std::string title, 
			  int nchX, double lowX, double highX)
             {return book1D(name, title, nchX, lowX, highX, fCurrentFolder);}
  /// book 1D variable bin histogram
  MonitorElement * book1D(std::string name, std::string title,
			  int nchX, float *xbinsize)
             {return book1D(name, title, nchX, xbinsize, fCurrentFolder);}

  /// book 2D histogram
  MonitorElement * book2D(std::string name, std::string title,
			  int nchX, double lowX, double highX, int nchY, 
			  double lowY, double highY)
             {return book2D(name, title, nchX, lowX, highX, 
	                                 nchY, lowY, highY, fCurrentFolder);}
  /// book 3D histogram
  MonitorElement * book3D(std::string name, std::string title,
			  int nchX, double lowX, double highX, int nchY, 
			  double lowY, double highY, int nchZ,
			  double lowZ, double highZ)
             {return book3D(name, title, nchX, lowX, highX, 
	                                 nchY, lowY, highY,
		                         nchZ, lowZ, highZ, fCurrentFolder);}
  /// book profile;
  /// option is one of: " ", "s" (default), "i", "G" (see TProfile::BuildOptions)
  /// (in a profile plot the number of channels in Y is disregarded)
  MonitorElement * bookProfile(std::string name, 
			       std::string title,int nchX, double lowX,
			       double highX, int nchY, double lowY, 
			       double highY, char * option = "s")
             {return bookProfile(name, title, nchX, lowX, highX, 
	                                      nchY, lowY, highY, fCurrentFolder, option);}

  /// book 2-D profile;
  /// option is one of: " ", "s" (default), "i", "G" (see TProfile2D::BuildOptions)
  /// (in a 2-D profile plot the number of channels in Z is disregarded)
  MonitorElement * bookProfile2D(std::string name, 
				 std::string title, 
				 int nchX, double lowX, double highX, 
				 int nchY, double lowY,double highY,
				 int nchZ, double lowZ,double highZ,
				 char * option = "s")
              {return bookProfile2D(name, title, nchX, lowX, highX, 
	                                         nchY, lowY, highY, 
						 nchZ, lowZ, highZ, fCurrentFolder, option);}

  /// book float
  MonitorElement * bookFloat(std::string s)
              {return bookFloat(s, fCurrentFolder);}
  /// book int
  MonitorElement * bookInt(std::string s)
              {return bookInt(s, fCurrentFolder);}
  /// book string
  MonitorElement * bookString(std::string s, std::string v)
              {return bookString(s, v, fCurrentFolder);}

//-------------------------------------------------------------------------
// ---------------------- public tagging ----------------------------------
  /// tag ME as <myTag> (myTag > 0)
  void tag(MonitorElement * me, unsigned int myTag);
  /// opposite action of tag method (myTag > 0)
  void untag(MonitorElement * me, unsigned int myTag);
  /// tag ME specified by full pathname (e.g. "my/long/dir/my_histo")
  void tag(std::string fullpathname, unsigned int myTag);
  /// opposite action of tag method
  void untag(std::string fullpathname, unsigned int myTag);
  /// tag all children of folder (does NOT include subfolders)
  void tagContents(std::string pathname,unsigned int myTag);
  /// opposite action of tagContents method
  void untagContents(std::string pathname,unsigned int myTag);
  /// tag all children of folder, including all subfolders and their children;
  /// exact pathname: FAST
  /// pathname including wildcards (*, ?): SLOW!
  void tagAllContents(std::string pathname, unsigned int myTag);
  /// opposite action of tagAllContents method
  void untagAllContents(std::string pathname, unsigned int myTag);

//-------------------------------------------------------------------------
// ---------------------- public ME getters -------------------------------

  /// get ME from full pathname (e.g. "my/long/dir/my_histo")
  MonitorElement * get(std::string fullpath) const;
  /// get all MonitorElements tagged as <tag>
  std::vector<MonitorElement *> get(unsigned int tag) const;
  /// get vector with all children of folder
  /// (does NOT include contents of subfolders)
  std::vector<MonitorElement *> getContents(std::string pathname) const;
  /// same as above for tagged MonitorElements
  std::vector<MonitorElement *> getContents
       (std::string pathname, unsigned int tag) const;
  /// get contents;
  /// return vector<string> of the form <dir pathname>:<obj1>,<obj2>,<obj3>;
  /// if showContents = false, change form to <dir pathname>:
  /// (useful for subscription requests; meant to imply "all contents")
  void getContents(std::vector<std::string> & put_here, 
		   bool showContents = true) const;

// ---------------------- temporarily public for Ecal/Hcal/ -------------
  /// reset contents (does not erase contents permanently)
  /// (makes copy of current contents; will be subtracted from future contents)
  void softReset(MonitorElement * me);

// ---------------------- Public deleting ---------------------------------
  /// remove directory
  void rmdir(std::string fullpath);
  /// erase all monitoring elements in current directory (not including subfolders);
  void removeContents(void) {removeContents(fCurrentFolder);}
  /// erase monitoring element in current directory 
  /// (opposite of book1D,2D,etc. action);
  void removeElement(std::string name) {removeElement(fCurrentFolder, name);}
  /// remove Monitor Element <name> from <pathname> in <Dir>
  void remove(const std::string & pathname, const std::string & name,
		  dqm::me_util::rootDir & Dir);

//-------------------------------------------------------------------------
// ---------------------- public I/O --------------------------------------

  /// save directory with monitoring objects into root file <filename>;
  /// include quality test results with status >= minimum_status 
  /// (defined in Core/interface/QTestStatus.h);
  /// if directory="", save full monitoring structure
  void save(std::string filename, std::string dir_fullpath="",
	    int minimum_status=dqm::qstatus::STATUS_OK);
  /// open/read root file <filename>, and copy MonitorElements;
  /// if flag=true, overwrite identical MonitorElements (default: false);
  /// if directory != "", read only selected directory
  /// if prepend !="", prepend string to path
  void open(std::string filename, bool overwrite = false,
	    std::string directory="",std::string prepend="");
  /// version info
  std::string getFileReleaseVersion(std::string filename);
  std::string getFileDQMPatchVersion(std::string filename);
  std::string getDQMPatchVersion() { return "DQMPATCH:"+dqmPatchVersion; } 

//-------------------------------------------------------------------------
// ---------------------- Public print methods -----------------------------
  void showDirStructure(void) const;

//-------------------------------------------------------------------------
// ---------------------- Quality Test methods -----------------------------
  /// get QCriterion corresponding to <qtname> 
  /// (null pointer if QCriterion does not exist)
  QCriterion * getQCriterion(std::string qtname) const;
  /// create quality test with unique name <qtname> (analogous to ME name);
  /// quality test can then be attached to ME with useQTest method
  /// (<algo_name> must match one of known algorithms)
  QCriterion * createQTest(std::string algo_name, std::string qtname);

  /// attach quality test <qc> to directory contents ==> FAST
  /// if tag != 0, this applies to tagged contents
  /// (need exact pathname without wildcards, e.g. A/B/C);
  ///
  void useQTest(std::string search_string, std::string qtname) const;
  /// attach quality test <qc> to all ME matching <search_string>;
  /// if tag != 0, this applies to tagged contents
  /// <search_string> could : (a) be exact pathname (e.g. A/B/C/histo): FAST
  /// (b) include wildcards (e.g. A/?/C/histo, A/B/\*/histo or A/B/\*): SLOW
  void useQTest(unsigned int tag, std::string search_string, 
		const dqm::me_util::rootDir & Dir, QCriterion * qc) const;

  /// get QReport from ME (null pointer if no such QReport)
  QReport * getQReport(MonitorElement * me, std::string qtname);

  /// run quality tests (also finds updated contents in last monitoring cycle,
  /// including newly added content) 
  void runQTests(void);

  /// get "global" folder <inpath> status (one of:STATUS_OK, WARNING, ERROR, OTHER);
  /// returns most sever error, where ERROR > WARNING > OTHER > STATUS_OK;
  /// see Core/interface/QTestStatus.h for details on "OTHER" 
  int getStatus(std::string inpath = "") const;

 private:

 
  void reParseConfig(const edm::ParameterSet &pset){}
 
  // ------------ Operations for MEs that are normally never reset ---------
  /// reverts action of softReset
  void disableSoftReset(MonitorElement * me);

  /// acquire and release lock
  void lock();
  void unlock();

  
  // ---------------- Navigation -----------------------
  

  /// Use this for saving monitoring objects in ROOT files with dir structure;
  /// cd into directory (create first if it doesn't exist);
  /// returns success flag
  bool cdInto(std::string inpath) const;

  /// name of global monitoring folder (containing all sources subdirectories)
  static const std::string monitorDirName;
  static const std::string referenceDirName;
  static const std::string collateDirName;
  static const std::string dqmPatchVersion;
  // ---------------- Miscellaneous -----------------------------
  

  // ------------------- Reference ME -------------------------------
  
  /// reference histogram (from file) 
  void             readReferenceME(std::string filename); // read from file
  bool             makeReferenceME(MonitorElement* me );  // copy into Referencedir
  bool             isReferenceME(MonitorElement* me) const; // check for existing
  bool             isCollateME(MonitorElement* me) const; // check for existing
  MonitorElement * getReferenceME(MonitorElement * me) const; // refme for given me
  void             deleteME(MonitorElement *me) ; // delete ME (for all me)


  // ---------------------------------------------------------------
  

  /// get verbose level
  unsigned getVerbose(void) const {return DQM_VERBOSE;}


  /// remove all references for directories starting w/ pathname;
  /// put contents of directories in removeContents (if applicable)
  void removeReferences(std::string pathname, dqm::me_util::rootDir & rDir);

  /// add <name> of folder to private method removedContents
  void add2RemovedContents(const MonitorElementRootFolder * subdir, 
			   std::string name);
  /// add contents of folder to private method removedContents
  void add2RemovedContents(const MonitorElementRootFolder * subdir);




  // ------------------- Private "getters" ------------------------------
  
  /// to be called by getContents (flag = false) or getMonitorable (flag = true)
  /// return vector<string> of the form <dir pathname>:<obj1>,<obj2>,<obj3>;
  /// if showContents = false, change form to <dir pathname>:
  /// (useful for subscription requests; meant to imply "all contents")
  void get(std::vector<std::string> & put_here, bool monit_flag, 
	   bool showContents = true) const;
  /// get first non-null ME found starting in path; null if failure
  MonitorElement * getMEfromFolder(dqm::me_util::cdir_it & path) const;
  /// get first ME found following obj_name in path; null if failure
  MonitorElement * getMEfromFolder(dqm::me_util::cdir_it & path, 
				   std::string obj_name) const;

  /// get monitorable;
  /// return vector<string> of the form <dir pathname>:<obj1>,<obj2>,<obj3>;
  /// if showContents = false, change form to <dir pathname>:
  /// (useful for subscription requests; meant to imply "all contents")
  void getMonitorable(std::vector<std::string> & put_here,
		      bool showContents = true) const;

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
  MonitorElementRootFolder * 
    makeDirectory(std::string inpath, dqm::me_util::rootDir & Dir);
    
  /// get (pointer to) last directory in inpath (null if inpath does not exist)
  MonitorElementRootFolder * getDirectory
    (std::string inpath, const dqm::me_util::rootDir& Dir) const;

  /// get folder corresponding to inpath wrt to root (create subdirs if necessary)
  MonitorElementRootFolder * makeDirectory(std::string inpath)
  {return makeDirectory(inpath, Own);}
  /// get (pointer to) last directory in inpath (null if inpath does not exist)
  MonitorElementRootFolder * getDirectory(std::string inpath) const
  {return getDirectory(inpath, Own);}

  /// read ROOT objects from file <file> in directory <pathname>;
  /// return total # of ROOT objects read
  unsigned int readDirectory(TFile * file, std::string pathname, std::string prepend = "");
  
  /// get MonitorElement <name> in directory <pathname>
  /// (null if MonitorElement does not exist)
  MonitorElement * findObject(std::string name, std::string pathname) const;

  /// look for folder <name> in current directory
  MonitorElementRootFolder * findFolder(std::string name) const
  {return fCurrentFolder->findFolder(name);}

  /// get root folder
  MonitorElementRootFolder * 
    getRootFolder(const dqm::me_util::rootDir & Dir) const{return Dir.top;}

  /// add all (tagged) MEs to put_here
  void get(const dqm::me_util::dir_map & Dir, 
	   std::vector<MonitorElement *> & put_here) const;
  /** get tags for various maps, return vector with strings of the form
      <dir pathname>:<obj1>/<tag1>/<tag2>,<obj2>/<tag1>/<tag3>, etc. */

  void getAllTags(std::vector<std::string> & put_here) const;
  void getAddedTags(std::vector<std::string> & put_here) const;
  void getRemovedTags(std::vector<std::string> & put_here) const;

  /// get vector with all children of folder in <rDir>
  /// (does NOT include contents of subfolders)
  void getContents(std::string & pathname, 
		   const dqm::me_util::rootDir & rDir,
		   std::vector<MonitorElement *> & put_here) const;

  /// get vector with children of folder, including all subfolders + their children;
  /// exact pathname: FAST
  /// pathname including wildcards (*, ?): SLOW!
  std::vector<MonitorElement*> getAllContents(std::string pathname) const;
  /// same as above for tagged MonitorElements
  std::vector<MonitorElement*> getAllContents(std::string pathname,
					      unsigned int tag) const;
  
  /// get vector with all children of folder and all subfolders of <rDir>;
  /// pathname may include wildcards (*, ?) ==> SLOW!
  void getAllContents (std::string & pathname, 
     const dqm::me_util::rootDir & rDir,
     std::vector<MonitorElement*> & put_here) const;
  
  /// get rootDir corresponding to tag 
  /// (Own for tag=0, or null for non-existing tag)
  const dqm::me_util::rootDir * getRootDir(unsigned int tag) const;

  /// check if added contents match rules; put matches in put_here
  void checkAddedContents(const dqm::me_util::searchCriteria & rules,
                          std::vector<MonitorElement *> & put_here) const;
  /// same as base class for given search_string and path; 
  /// put matches into put_here
  void checkAddedContents(const std::string & search_string, 
			  dqm::me_util::cmonit_it & added_path,
			  const dqm::me_util::rootDir & Dir,
			  std::vector<MonitorElement*> & put_here) const;


  /// check if added contents match search paths
  void checkAddedSearchPaths
    (const std::vector<std::string> & search_path, 
     const dqm::me_util::rootDir & Dir,
     std::vector<MonitorElement*>& put_here) const;

  /// check if added contents belong to folders 
  /// (use flag to specify if subfolders should be included)
  void checkAddedFolders(const std::vector<std::string> & folders,
                         const dqm::me_util::rootDir & Dir,
                         bool useSubfolders,
                         std::vector<MonitorElement*>& put_here) const;

  /// check if added contents belong to folder 
  /// (use flag to specify if subfolders should be included)
  void checkAddedFolder(dqm::me_util::cmonit_it & added_path,
			const dqm::me_util::rootDir & Dir,
			std::vector<MonitorElement*>& put_here) const;

  /// check if added contents are tagged
  void checkAddedTags(const dqm::me_util::rootDir & Dir,
		      std::vector<MonitorElement*>& put_here) const;

  /// make copies for <me> for all tags it comes with
  /// (to be called the first time <me> is received or read from a file)
  void makeTagCopies(MonitorElement * me);

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
  bool extractObject(TObject * to, MonitorElementRootFolder * dir, 
		     bool fromRemoteNode);
  /// extract object (TH1F, TH2F, ...) from <to>; return success flag
  /// flag fromRemoteNode indicating if ME arrived from different node
  bool extractObject(TObject * to, bool fromRemoteNode,
		     std::string & name, std::string & value);
  /// extract TH1F object from <to> in <dir>; 
  /// flag fromRemoteNode indicating if ME arrived from different node
  void extractTH1F(TObject * to, MonitorElementRootFolder * dir, 
		   bool fromRemoteNode);
  /// extract TH2F object from <to> in <dir>; 
  /// flag fromRemoteNode indicating if ME arrived from different node
  void extractTH2F(TObject * to, MonitorElementRootFolder * dir, 
		   bool fromRemoteNode);
  /// extract TH3F object from <to> in <dir>; 
  /// flag fromRemoteNode indicating if ME arrived from different node
  void extractTH3F(TObject * to, MonitorElementRootFolder * dir, 
		   bool fromRemoteNode);
  /// extract TProfile object from <to> in <dir>; 
  /// flag fromRemoteNode indicating if ME arrived from different node
  void extractTProf(TObject * to, MonitorElementRootFolder * dir, 
		    bool fromRemoteNode);
  /// extract TProfile2D object from <to> in <dir>; 
  /// flag fromRemoteNode indicating if ME arrived from different node
  void extractTProf2D(TObject * to, MonitorElementRootFolder * dir, 
		      bool fromRemoteNode);
  /// extract integer from <to> in <dir>; 
  /// flag fromRemoteNode indicating if ME arrived from different node
  void extractInt(TObject * to, MonitorElementRootFolder * dir, 
		  bool fromRemoteNode);
  /// extract float from <to> in <dir>; 
  /// flag fromRemoteNode indicating if ME arrived from different node
  void extractFloat(TObject * to, MonitorElementRootFolder * dir, 
		    bool fromRemoteNode);
  /// extract string from <to> in <dir>; 
  /// flag fromRemoteNode indicating if ME arrived from different node
  void extractString(TObject * to, MonitorElementRootFolder * dir, 
		     bool fromRemoteNode);
  /// extract QReport 
  void extractQReport(TObject * to, MonitorElementRootFolder * dir,
		      bool fromRemoteNode);

  /// true if TObject is a TH1F
  inline bool isTH1F(const TObject * to) const {
    TString ctype = to->IsA()->GetName(); return ( ctype(0,4) == "TH1F");
  }
  /// true if TObject is a TH2F
  inline bool isTH2F(const TObject * to) const {
    TString ctype = to->IsA()->GetName(); return ( ctype(0,4) == "TH2F");
  }
  /// true if TObject is a TH3F
  inline bool isTH3F(const TObject * to) const {
    TString ctype = to->IsA()->GetName(); return ( ctype(0,4) == "TH3F");
  }
  /// true if TObject is a TProfile2D
  inline bool isTProf2D(const TObject * to) const {
    TString ctype = to->IsA()->GetName(); return ( ctype(0,10) == "TProfile2D");
  }
  /// true if TObject is a TProfile
  inline bool isTProf(const TObject * to) const {
    TString ctype = to->IsA()->GetName(); return ( ctype(0,8) == "TProfile");
  }
  /// true if TObject is a TObjString corresponding to an integer
  inline bool isInt(const TObject * to) const {
    return hasKey(to, "i=");
  }
  /// true if TObject is a TObjString corresponding to a float
  inline bool isFloat(const TObject * to) const {
    return hasKey(to, "f=");
  }
  /// true if TObject is a TObjString corresponding to a string
  inline bool isString(const TObject * to) const {
    return hasKey(to, "s=");
  }
  /// true if TObject is a TObjString corresponding to a quality report
  inline bool isQReport(const TObject * to) const {
    return hasKey(to, "qr=");
  }
  /** true if TObject is (a) either TObjString with ">"+key in name 
      (b) or TNamed with "key" in title; 
      to be used for int, float, string, qreport */
  inline bool hasKey(const TObject * to, const std::string & key) const {
    TString ctype = to->IsA()->GetName();
    if(ctype(0,5) == "TObjS") { // for remote nodes (TObjString * getTagObject)
	std::string nm(to->GetName());
	return (nm.find(">"+key) != std::string::npos);
    }
    if(ctype(0,6) == "TNamed") { // for reading from files (TNamed *getRootObject)
	std::string nm(to->GetTitle());
	return (nm.find(key) != std::string::npos);
    }
    return false;
  }

  /// true if ME should be extracted from object;
  /// for remoteNode: true if replacing old ME or ME is desired
  /// for local ME: true if ME does not exist or we can overwrite
  bool wantME(MonitorElement * me, MonitorElementRootFolder * dir, 
	      const std::string & nm, bool fromRemoteNode) const;

  /// true if Monitoring Element <me> in directory <folder> has isDesired = true;
  /// if warning = true and <me> does not exist, show warning
  bool isDesired(MonitorElementRootFolder * folder, 
		 std::string me, bool warning) const;
 
  /// unpack TObjString into name <nm> and value <value>; return success
  /// (for remote nodes: TObjString * getTagObject)
  bool unpack(TObjString * tn, std::string & nm, std::string & value) const;
  /** unpack TNamed into name <nm> and value <value>; return success
      for reading from files: TNamed *getRootObject */
  bool unpack(TNamed * tn, std::string & nm, std::string & value) const;
  // ------------------- Booking ---------------------------

  /// add monitoring element to directory <pathname>
  void addElement(MonitorElement * me, std::string pathname, 
		  std::string type = "");
  /// add null monitoring element to folder <pathname> (can NOT be folder);
  /// used for registering monitorables before user has subscribed to <name>
  void addElement(std::string name, std::string pathname);


  // ---------------------- Booking ------------------------------------
  //book 1D histogram using existing histogram
  MonitorElement * clone1D(std::string name, TH1F* source)
                            {return book1D(name,source,fCurrentFolder);}

  //book 2D histogram using existing histogram
  MonitorElement * clone2D(std::string name, TH2F* source)
  {return book2D(name,source,fCurrentFolder);}

  //book 3D histogram using existing histogram
  MonitorElement * clone3D(std::string name, TH3F* source)
  {return book3D(name,source,fCurrentFolder);}

  //book TProfile using existing profile
  MonitorElement * cloneProfile(std::string name, TProfile* source)
  {return bookProfile(name,source,fCurrentFolder);}

  //book TProfile2D using existing profile
  MonitorElement * cloneProfile2D(std::string name, TProfile2D* source)
  {return bookProfile2D(name,source,fCurrentFolder);}

  ///  book 1D histogram based on TH1F
  MonitorElement * book1D(std::string name, TH1F* source,
                           MonitorElementRootFolder * dir);
  /// same method with different name
  //MonitorElement * clone1D(std::string name, TH1F* source,
  //                            MonitorElementRootFolder * dir)
  //     {return book1D(name, source, dir);}
       
  /// book 1D histogram based on bin defs.
  MonitorElement * book1D(std::string name, std::string title, 
			  int nchX, double lowX, double highX, 
			  MonitorElementRootFolder * folder);
  /// book 1D variable bin histogram
  MonitorElement * book1D(std::string name, std::string title,
			  int nchX, float *xbinsize,
			  MonitorElementRootFolder * folder);


  ///  book 2D histogram based on TH2F
  MonitorElement * book2D(std::string name, TH2F* source,
                              MonitorElementRootFolder * dir);
  /// same method with different name
  //MonitorElement * clone2D(std::string name, TH2F* source,
  //                            MonitorElementRootFolder * dir)
  //     {return book2D(name, source, dir);}
  /// book 2D histogram
  MonitorElement * book2D(std::string name, std::string title,
			  int nchX, double lowX, double highX, int nchY, 
			  double lowY, double highY, 
			  MonitorElementRootFolder * folder);

  ///  book 3D histogram based on TH3F
  MonitorElement * book3D(std::string name, TH3F* source,
                              MonitorElementRootFolder * dir);
  /// same method with different name
  //MonitorElement * clone3D(std::string name, TH3F* source,
  //                            MonitorElementRootFolder * dir)
  //     {return book3D(name, source, dir);}
  /// book 3D histogram
  MonitorElement * book3D(std::string name, std::string title,
			  int nchX, double lowX, double highX, int nchY, 
			  double lowY, double highY, int nchZ,
			  double lowZ, double highZ, 
			  MonitorElementRootFolder * folder);

  ///  book profile histogram based on TProfile
  MonitorElement * bookProfile(std::string name, TProfile* source,
                              MonitorElementRootFolder * dir);
  /// same method with different name
  //MonitorElement * cloneProfile(std::string name, TProfile* source,
  //                            MonitorElementRootFolder * dir)
  //     {return bookProfile(name, source, dir);}
  /// book profile;
  /// option is one of: " ", "s" (default), "i", "G" (see TProfile::BuildOptions)
  /// (in a profile plot the number of channels in Y is disregarded)
  MonitorElement * bookProfile(std::string name, 
			       std::string title,int nchX, double lowX,
			       double highX, int nchY, double lowY, 
			       double highY, MonitorElementRootFolder* folder,
			       char * option = "s");

  /// book 2-D profile based on TProfile2D
  MonitorElement * bookProfile2D(std::string name, TProfile2D* source,
				 MonitorElementRootFolder * dir);
  /// same method with different name
  //MonitorElement * cloneProfile2D(std::string name, TProfile2D* source,
  //                            MonitorElementRootFolder * dir)
  //     {return bookProfile2D(name, source, dir);}
  /// option is one of: " ", "s" (default), "i", "G" (see TProfile2D::BuildOptions)
  /// (in a 2-D profile plot the number of channels in Z is disregarded)
  MonitorElement * bookProfile2D(std::string name, 
				 std::string title, 
				 int nchX, double lowX, double highX, 
				 int nchY, double lowY,double highY,
				 int nchZ, double lowZ,double highZ,
				 MonitorElementRootFolder * folder, 
				 char * option="s");

  /// book float
  MonitorElement * bookFloat(std::string s, 
			     MonitorElementRootFolder * folder);
  /// book int
  MonitorElement * bookInt(std::string s, 
			   MonitorElementRootFolder * folder);
  /// book string
  MonitorElement * bookString(std::string s, std::string v, 
			      MonitorElementRootFolder * folder);

  // ---------------- Checkers -----------------------------
  
  /// true if pathname exists
  bool pathExists(std::string inpath,
		  const dqm::me_util::rootDir & Dir) const;
  /// check against null objects (true if object exists)
  bool checkElement(const MonitorElement * const me) const;
  

  /// true if Monitoring Element <me> is needed by any subscriber
  bool isNeeded(std::string pathname, std::string me) const;

  /// -------------------- Unsubscribing/Removing --------------------

  /// remove monitoring element from directory; 
  /// if warning = true, print message if element does not exist
  void removeElement(MonitorElementRootFolder* dir, std::string name,
		     bool warning = true);
  /// remove all monitoring elements from directory; 
  void removeContents(MonitorElementRootFolder * dir);
  
  /// remove all contents from <pathname> from all subscribers, tags and CMEs
  void removeCopies(const std::string & pathname);
  /// remove Monitor Element <name> from all subscribers, tags & CME directories
  void removeCopies(const std::string & pathname, const std::string & name);
  // -------------------- Deleting ----------------------------------
  /// delete directory and all contents;
  /// delete directory (all contents + subfolders);
  void rmdir(const std::string & pathname, dqm::me_util::rootDir & rDir);

  /// copy monitoring elements from source to destination
  void copy(const MonitorElementRootFolder * const source, 
	    MonitorElementRootFolder * const dest, 
	    const std::vector<std::string> & contents);
  /// remove subscribed monitoring elements; 
  /// if warning = true, printout error messages when problems
  void removeSubsc(MonitorElementRootFolder* const dir, 
		   const std::vector<std::string> & contents, 
		   bool warning = true);
  
  // -------------------- Misc ----------------------------------

  /// update directory structure maps for folder
  void updateMaps(MonitorElementRootFolder* dir, dqm::me_util::rootDir & rDir);

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

  /// same as above for tag;
  int getStatus(unsigned int tag) const;
  /// same as above for vector with MonitorElements
  int getStatus(const std::vector<MonitorElement *> & ME_group) const;

  /// true if at least one ME gave hasError/hasWarning/hasOtherReport = true
  bool hasError(const std::vector<MonitorElement *> & ME_group) const;
  bool hasWarning(const std::vector<MonitorElement *> & ME_group) const;
  bool hasOtherReport(const std::vector<MonitorElement *> & ME_group) const;
  

  // --- Operations on MEs that are normally reset at end of monitoring cycle ---

  /// if true, will accumulate ME contents (over many periods)
  /// until method is called with flag = false again
  void setAccumulate(MonitorElement * me, bool flag);

  /// universal verbose flag for DQM
  unsigned DQM_VERBOSE;

  // -------------------- Quality tests on MonitorElements ------------------

   /// add quality report (to be called when test is to run locally)
  QReport * addQReport(MonitorElement * me, QCriterion * qc) const;
  /// add quality report (to be called by ReceiverBase)
  QReport * addQReport(MonitorElement * me, std::string qtname, 
		       QCriterion * qc = 0) const;
  /// add quality report to ME
  void addQReport(MonitorElement * me, QReport * qr) const
  {me->addQReport(qr);}

  /// add quality reports to all MEs
  void addQReport
    (std::vector<MonitorElement *> & allMEs, QCriterion * qc) const;

  /// check if QReport is already defined for ME
  bool qreportExists(MonitorElement * me, std::string qtname) const
  {return me->qreportExists(qtname);}



  /// set of all available algorithms for quality tests
  std::set<std::string> availableAlgorithms;




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
  void scanContents(const std::string & search_string, 
		    const MonitorElementRootFolder * folder,
		    std::vector<MonitorElement *> & put_here) const;

  /// look for all MEs matching <search_string> in Own;
  /// if found, create QReport from QCriterion and add to ME
  void scanContents(QCriterion * qc, const std::string & search_string) const;

  /// check if resetMonitoringDiff and resetMonitorableDiff were called 
  /// (to be reset in MonitorUserInterface::runQualityTests)
  inline bool wasResetCalled() const
  {return rMonitoringDiffWasCalled && rMonitorableDiffWasCalled;}

 





 //-------------------------------------------------------------------------------
 //-------------------------------------------------------------------------------
 
 
  /// use to printout warning when calling quality tests twice without
  /// having called resetMonitoringDiff, resetMonitorableDiff in between...
  /// (to be reset in MonitorUserInterface::runQualityTests)
  bool rMonitoringDiffWasCalled;
  bool rMonitorableDiffWasCalled;
  bool overwriteFromFile;
  bool first_time_onRoot;

  /// true if fCurrentFolder is the root folder (gROOT->GetRootFolder() )
  bool isRootFolder(void);
  
  /// if non-empty, read from file only selected directory
  std::string readOnlyDirectory;
  
  /// run quality tests (also finds updated contents in last monitoring cycle,
  /// including newly added content) <-- to be called only by runQTests
  void runQualityTests(void);

  /// loop over quality tests & addedContents: look for MEs that 
  /// match QCriterion::rules; upon a match, add QReport to ME(s)
  void checkAddedElements(void);
  //
  DaqMonitorBEInterface(const DaqMonitorBEInterface&);
  const DaqMonitorBEInterface& operator=(const DaqMonitorBEInterface&);

  /// make new directory structure for Subscribers, Tags and CMEsx
  void makeDirStructure(dqm::me_util::rootDir & Dir, std::string name);

  // ------ these were friends before
  friend class edm::DQMHttpSource;   // In EventFilter/StorageManager

  friend class MonitorUserInterface;
  friend class MonitorUIRoot;
  friend class DQMTagHelper;

  friend class ClientRoot;             // these will likely be removed
  friend class ClientServerRoot;
  friend class NodeBase;
  friend class SenderBase;
  friend class ReceiverBase;

  // ------ these friends from within DQMServices added since V01-00-01
  friend class RootMonitorThread;
  friend class QTestStatusChecker;

  friend class ROOTtoMEConverter;      // need clone methods
  friend class MEtoROOTConverter;

  // ------ example executables        // can all be removed ???
  friend class DQMLocalGUI;
  friend class DQMBackEndInterfaceExample;
  friend class DQMMonitorUIRootStandaloneExample;
  friend class DQMReadFileExample;
  friend class DQMBackEndInterfaceQTestsExample;
  
  // ------ old web GUI class using lock and unlock
  friend class WebInterface;           // 
  friend class ClientWithWebInterface; // needs getUpdatedContents
  friend class MuonDQMClient;          // needs getUpdatedContents
  friend class SiPixelWebClient;       // needs getUpdatedContents
        
  // ----------------------- private data members
  MonitorElementRootFolder * fCurrentFolder;
  DQMTagHelper * tagHelper;

  // ----------------------- singleton admin -----------------------------------
  static DaqMonitorBEInterface *instance();
  static DaqMonitorBEInterface *theinstance;

};


#endif
