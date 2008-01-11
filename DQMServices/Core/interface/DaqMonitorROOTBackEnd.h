/*
 * $Date: 2008/01/01 21:17:34 $
 * $Revision: 1.32 $
 * $Author: elmer $
*/

#ifndef DaqMonitorROOTBackEnd_h
#define DaqMonitorROOTBackEnd_h

#include <vector>
#include <set>
#include <map>
#include <string>

#include "TCanvas.h"
#include "TObject.h"
#include "TClass.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Core/interface/MonitorElementRootT.h"

// forward declaration of abstract interfaces
class MonitorElement;
class QReport;
class QCriterion;
class DQMTagHelper;
class TFile;

namespace edm {
  class DQMHttpSource;
}

class DaqMonitorROOTBackEnd : public DaqMonitorBEInterface
{
  
 public:
  
  DaqMonitorROOTBackEnd(edm::ParameterSet const&);  
  
  virtual ~DaqMonitorROOTBackEnd();
  
  /* ---------------------- Booking ------------------------------------ */
    
  /// other methods of book 1D
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
  {return book2D(name, title, nchX, lowX, highX, nchY, lowY, highY, 
		  fCurrentFolder);}
  /// book 3D histogram
  MonitorElement * book3D(std::string name, std::string title,
			  int nchX, double lowX, double highX, int nchY, 
			  double lowY, double highY, int nchZ,
			  double lowZ, double highZ)
  {return book3D(name, title, nchX, lowX, highX, nchY, lowY, highY,
		  nchZ, lowZ, highZ, fCurrentFolder);}
  /// book profile;
  /// option is one of: " ", "s" (default), "i", "G" (see TProfile::BuildOptions)
  /// (in a profile plot the number of channels in Y is disregarded)
  MonitorElement * bookProfile(std::string name, 
			       std::string title,int nchX, double lowX,
			       double highX, int nchY, double lowY, 
			       double highY, char * option = "s")
  {return bookProfile(name, title, nchX, lowX, highX, nchY, lowY, highY,
		      fCurrentFolder, option);}
  /// book 2-D profile;
  /// option is one of: " ", "s" (default), "i", "G" (see TProfile2D::BuildOptions)
  /// (in a 2-D profile plot the number of channels in Z is disregarded)
  MonitorElement * bookProfile2D(std::string name, 
				 std::string title, 
				 int nchX, double lowX, double highX, 
				 int nchY, double lowY,double highY,
				 int nchZ, double lowZ,double highZ,
				 char * option = "s")
  {return bookProfile2D(name, title, nchX, lowX, highX, nchY, lowY, 
			highY, nchZ, lowZ, highZ, fCurrentFolder,
			option);}

  /// book float
  MonitorElement * bookFloat(std::string s)
  {return bookFloat(s, fCurrentFolder);}
  /// book int
  MonitorElement * bookInt(std::string s)
  {return bookInt(s, fCurrentFolder);}
  /// book string
  MonitorElement * bookString(std::string s, std::string v)
  {return bookString(s, v, fCurrentFolder);}
  
  /// ---------------- Navigation -----------------------
 
  /// retun pathname of current directory
  std::string pwd(void) const {return fCurrentFolder->getPathname();}
  /// go to top directory (ie. root)
  void cd(void) {setCurrentFolder("/");}
  /// equivalent to "cd .."
  void goUp(void);
  /// set the last directory in fullpath as the current directory(create if needed);
  /// to be invoked by user to specify directories for monitoring objects 
  /// before booking;
  /// commands book1D (etc) & removeElement(name) imply elements in this directory!;
    void setCurrentFolder(std::string fullpath);
    /// cd to subdirectory (if there)
  void cd(std::string subdir_path);

  /// ---------------- Miscellaneous -----------------------------

  /// show directory structure
  void showDirStructure(void) const;
  /** save dir_fullpath with monitoring objects into root file <filename>;
      include quality test results with status >= minimum_status 
      (defined in Core/interface/QTestStatus.h);
      if dir_fullpath="", save full monitoring structure */
  void save(std::string filename, std::string dir_fullpath="",
	    int minimum_status=dqm::qstatus::STATUS_OK);
  /// open/read root file <filename>, and copy MonitorElements;
  /// if flag=true, overwrite identical MonitorElements (default: false);
  /// if directory != "", read only selected directory
  /// if prepend !="", prepend string to path
  void open(std::string filename, bool overwrite = false,
	    std::string directory="",std::string prepend="");
  /// get list of subdirectories of current directory
  std::vector<std::string> getSubdirs(void) const;
  /// get list of (non-dir) MEs of current directory
  std::vector<std::string> getMEs(void) const;

  /// ------------- Tags (e.g. detector-IDs;) -----------------
  /// Property similar to "Labels" of google-mail) 

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

  /// ------------------- "Getters" ------------------------------

  /// get all contents;
  /// return vector<string> of the form <dir pathname>:<obj1>,<obj2>,<obj3>;
  /// if showContents = false, change form to <dir pathname>:
  /// (useful for subscription requests; meant to imply "all contents")
  void getContents(std::vector<std::string> & put_here, 
		   bool showContents = true) const;
  /// get monitorable;
  /// return vector<string> of the form <dir pathname>:<obj1>,<obj2>,<obj3>;
  /// if showContents = false, change form to <dir pathname>:
  /// (useful for subscription requests; meant to imply "all contents")
  void getMonitorable(std::vector<std::string> & put_here,
		      bool showContents = true) const;

  /// ------------------- Public "getters" ------------------------------
  /// get ME from full pathname (e.g. "my/long/dir/my_histo")
  MonitorElement * get(std::string fullpath) const;
  

  /// get MonitorElement <name> in directory <pathname>
  /// (null if MonitorElement does not exist)
  MonitorElement * findObject(std::string name, std::string pathname) const;

  /// get all MonitorElements tagged as <tag>
  std::vector<MonitorElement *> get(unsigned int tag) const;

  /// get vector with all children of folder
  /// (does NOT include contents of subfolders)
  std::vector<MonitorElement *> getContents(std::string pathname) const;
  /// same as above for tagged MonitorElements
  std::vector<MonitorElement *> getContents
    (std::string pathname, unsigned int tag) const;

  /// get vector with children of folder, including all subfolders + their children;
  /// exact pathname: FAST
  /// pathname including wildcards (*, ?): SLOW!
  std::vector<MonitorElement*> getAllContents(std::string pathname) const;
  /// same as above for tagged MonitorElements
  std::vector<MonitorElement*> getAllContents(std::string pathname,
					      unsigned int tag) const;
  
  /// -------------------- Deleting ----------------------------------

  /// remove directory
  void rmdir(std::string fullpath);
  /// erase monitoring element in current directory 
  /// (opposite of book1D,2D,etc. action);
  void removeElement(std::string name)
  {removeElement(fCurrentFolder, name);}

  /// erase all monitoring elements in current directory (not including subfolders);
  void removeContents(void)
  {removeContents(fCurrentFolder);}

 protected:
  static DaqMonitorBEInterface *instance();
    
 private:
  
  static DaqMonitorROOTBackEnd *theinstance;

  TCanvas *c1;
  /// object name, pathname of last ME plotted (used in drawAll method)
  std::string last_objname;
  std::string last_pathname;
  ///
  /// ROOT objects are saved here when they are booked (set by "setCurrentFolder")
  MonitorElementRootFolder * fCurrentFolder;

  /// flag for printing warning just once
  bool first_time_onRoot;
  /// flag for overwriting MEs when reading from file (default: false)
  bool overwriteFromFile;
  /// if non-empty, read from file only selected directory
  std::string readOnlyDirectory;
  /// ------------------- Private "getters" ------------------------------

  /// to be called by getContents (flag = false) or getMonitorable (flag = true)
  /// return vector<string> of the form <dir pathname>:<obj1>,<obj2>,<obj3>;
  /// if showContents = false, change form to <dir pathname>:
  /// (useful for subscription requests; meant to imply "all contents")
  void get(std::vector<std::string> & put_here, bool monit_flag, 
	   bool showContents = true) const;

  /// get root folder
  MonitorElementRootFolder * 
    getRootFolder(const dqm::me_util::rootDir & Dir) const{return Dir.top;}

  /// get folder corresponding to inpath wrt to root (create subdirs if necessary)
  MonitorElementRootFolder * 
    makeDirectory(std::string inpath, dqm::me_util::rootDir & Dir);
  /// get (pointer to) last directory in inpath (null if inpath does not exist)
  MonitorElementRootFolder * getDirectory
    (std::string inpath, const dqm::me_util::rootDir& Dir) const;

  /// look for object <name> in current directory
  ///  MonitorElement * findObject(std::string name) const
  ///{return fCurrentFolder->findObject(name);}

  /// look for folder <name> in current directory
  MonitorElementRootFolder * findFolder(std::string name) const
  {return fCurrentFolder->findFolder(name);}

  /// get first non-null ME found starting in path; null if failure
  MonitorElement * getMEfromFolder(dqm::me_util::cdir_it & path) const;
  /// get first ME found following obj_name in path; null if failure
  MonitorElement * getMEfromFolder(dqm::me_util::cdir_it & path, 
				   std::string obj_name) const;

  /// get vector with all children of folder in <rDir>
  /// (does NOT include contents of subfolders)
  void getContents(std::string & pathname, 
		   const dqm::me_util::rootDir & rDir,
		   std::vector<MonitorElement *> & put_here) const;

  /// get vector with all children of folder and all subfolders of <rDir>;
  /// pathname may include wildcards (*, ?) ==> SLOW!
  void getAllContents
    (std::string & pathname, const dqm::me_util::rootDir & rDir,
     std::vector<MonitorElement*> & put_here) const;

  /// add all (tagged) MEs to put_here
  void get(const dqm::me_util::dir_map & Dir, 
	   std::vector<MonitorElement *> & put_here) const;

  /** get tags for various maps, return vector with strings of the form
      <dir pathname>:<obj1>/<tag1>/<tag2>,<obj2>/<tag1>/<tag3>, etc. */
  void getAllTags(std::vector<std::string> & put_here) const;
  void getAddedTags(std::vector<std::string> & put_here) const;
  void getRemovedTags(std::vector<std::string> & put_here) const;
  
  /// check if added contents belong to folder 
  /// (use flag to specify if subfolders should be included)
  void checkAddedFolder(dqm::me_util::cmonit_it & added_path,
			const dqm::me_util::rootDir & Dir,
			std::vector<MonitorElement*>& put_here) const;
  /// check if added contents match search paths
  void checkAddedSearchPaths
    (const std::vector<std::string> & search_path, 
     const dqm::me_util::rootDir & Dir,
     std::vector<MonitorElement*>& put_here) const;
  
  /// same as base class for given search_string and path; 
  /// put matches into put_here
  void checkAddedContents(const std::string & search_string, 
			  dqm::me_util::cmonit_it & added_path,
			  const dqm::me_util::rootDir & Dir,
			  std::vector<MonitorElement*> & put_here) const;

  /// make copies for <me> for all tags it comes with
  /// (to be called the first time <me> is received or read from a file)
  void makeTagCopies(MonitorElement * me);
  /// unpack TObjString into name <nm> and value <value>; return success
  /// (for remote nodes: TObjString * getTagObject)
  bool unpack(TObjString * tn, std::string & nm, std::string & value) const;
  /** unpack TNamed into name <nm> and value <value>; return success
      for reading from files: TNamed *getRootObject */
  bool unpack(TNamed * tn, std::string & nm, std::string & value) const;
  /// extract object (either TObjString or TNamed depending on flag);
  /// return success
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

  /// extract object (TH1F, TH2F, ...) from <to>; return success flag
  /// flag fromRemoteNode indicating if ME arrived from different node
  bool extractObject(TObject * to, MonitorElementRootFolder * dir, 
		     bool fromRemoteNode);

  /// ---------------- Checkers -----------------------------

  /// true if pathname exists
  bool pathExists(std::string inpath,
		  const dqm::me_util::rootDir & Dir) const;
  /// true if fCurrentFolder is the root folder (gROOT->GetRootFolder() )
  bool isRootFolder(void);
  /// true if directory (or any subfolder at any level below it) contains
  /// at least one valid (i.e. non-null) monitoring element
  bool containsAnyMEs(std::string pathname) const;
  /// true if directory (or any subfolder at any level below it) contains
  /// at least one monitorable element
  bool containsAnyMonitorable(std::string pathname) const;
  /// true if Monitoring Element <me> is needed by any subscriber
  bool isNeeded(std::string pathname, std::string me) const;
  /// true if Monitoring Element <me> in directory <folder> has isDesired = true;
  /// if warning = true and <me> does not exist, show warning
  bool isDesired(MonitorElementRootFolder * folder, 
		 std::string me, bool warning) const;

  /// true if ME should be extracted from object;
  /// for remoteNode: true if replacing old ME or ME is desired
  /// for local ME: true if ME does not exist or we can overwrite
  bool wantME(MonitorElement * me, MonitorElementRootFolder * dir, 
	      const std::string & nm, bool fromRemoteNode) const;

  /// true if TObject is a TH1F
  inline bool isTH1F(const TObject * to) const
  {
    TString ctype = to->IsA()->GetName();
    return ( ctype(0,4) == "TH1F");
  }
  /// true if TObject is a TH2F
  inline bool isTH2F(const TObject * to) const
  {
    TString ctype = to->IsA()->GetName();
    return ( ctype(0,4) == "TH2F");
  }
  /// true if TObject is a TH3F
  inline bool isTH3F(const TObject * to) const
  {
    TString ctype = to->IsA()->GetName();
    return ( ctype(0,4) == "TH3F");
  }
  /// true if TObject is a TProfile2D
  inline bool isTProf2D(const TObject * to) const
  {
    TString ctype = to->IsA()->GetName();
    return ( ctype(0,10) == "TProfile2D");
  }
  /// true if TObject is a TProfile
  inline bool isTProf(const TObject * to) const
  {
    TString ctype = to->IsA()->GetName();
    return ( ctype(0,8) == "TProfile");
  }
  /** true if TObject is (a) either TObjString with ">"+key in name 
      (b) or TNamed with "key" in title; 
      to be used for int, float, string, qreport */
  inline bool hasKey(const TObject * to, const std::string & key) const
  {
    TString ctype = to->IsA()->GetName();
    if(ctype(0,5) == "TObjS")// for remote nodes (TObjString * getTagObject)
      {
	std::string nm(to->GetName());
	return (nm.find(">"+key) != std::string::npos);
      }
    if(ctype(0,6) == "TNamed")// for reading from files (TNamed *getRootObject)
      {
	std::string nm(to->GetTitle());
	return (nm.find(key) != std::string::npos);
      }
    return false;
  }
  /// true if TObject is a TObjString corresponding to an integer
  inline bool isInt(const TObject * to) const
  {
    return hasKey(to, "i=");
  }
  /// true if TObject is a TObjString corresponding to a float
  inline bool isFloat(const TObject * to) const
  {
    return hasKey(to, "f=");
  }
  /// true if TObject is a TObjString corresponding to a string
  inline bool isString(const TObject * to) const
  {
    return hasKey(to, "s=");
  }
  /// true if TObject is a TObjString corresponding to a quality report
  inline bool isQReport(const TObject * to) const
  {
    return hasKey(to, "qr=");
  }

  /// ---------------------- Booking ------------------------------

  /// add monitoring element to directory <pathname>
  void addElement(MonitorElement * me, std::string pathname, 
		  std::string type = "");
  /// add null monitoring element to folder <pathname> (can NOT be folder);
  /// used for registering monitorables before user has subscribed to <name>
  void addElement(std::string name, std::string pathname);

  ///  book 1D histogram based on TH1F
  MonitorElement * book1D(std::string name, TH1F* source,
                              MonitorElementRootFolder * dir);
  /// same method with different name
  MonitorElement * clone1D(std::string name, TH1F* source,
                              MonitorElementRootFolder * dir)
       {return book1D(name, source, dir);}
       
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
  MonitorElement * clone2D(std::string name, TH2F* source,
                              MonitorElementRootFolder * dir)
       {return book2D(name, source, dir);}
  /// book 2D histogram
  MonitorElement * book2D(std::string name, std::string title,
			  int nchX, double lowX, double highX, int nchY, 
			  double lowY, double highY, 
			  MonitorElementRootFolder * folder);

  ///  book 3D histogram based on TH3F
  MonitorElement * book3D(std::string name, TH3F* source,
                              MonitorElementRootFolder * dir);
  /// same method with different name
  MonitorElement * clone3D(std::string name, TH3F* source,
                              MonitorElementRootFolder * dir)
       {return book3D(name, source, dir);}
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
  MonitorElement * cloneProfile(std::string name, TProfile* source,
                              MonitorElementRootFolder * dir)
       {return bookProfile(name, source, dir);}
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
  MonitorElement * cloneProfile2D(std::string name, TProfile2D* source,
                              MonitorElementRootFolder * dir)
       {return bookProfile2D(name, source, dir);}
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

  /// ---------------- Miscellaneous -----------------------------
  ///
  /// update directory structure maps for folder
  void updateMaps(MonitorElementRootFolder* dir, dqm::me_util::rootDir & rDir);

  /// ---------------- Navigation -----------------------

  /// Use this for saving monitoring objects in ROOT files with dir structure;
  /// cd into directory (create first if it doesn't exist);
  /// returns success flag
  bool cdInto(std::string inpath) const;
  
  /// -------------------- Deleting ----------------------------------

  /// delete directory (all contents + subfolders);
  void rmdir(const std::string & pathname, dqm::me_util::rootDir & rDir);
  /// remove all references for directories starting w/ pathname;
  /// put contents of directories in removeContents (if applicable)
  void removeReferences(std::string pathname, dqm::me_util::rootDir & rDir);
  /// copy monitoring elements from source to destination
  void copy(const MonitorElementRootFolder * const source, 
	    MonitorElementRootFolder * const dest, 
	    const std::vector<std::string> & contents);
  /// remove subscribed monitoring elements;
  /// if warning = true, printout error messages when problems
  void removeSubsc(MonitorElementRootFolder* const dir, 
		   const std::vector<std::string> & contents, 
		   bool warning = true);
  /// add <name> of folder to private method removedContents
  void add2RemovedContents(const MonitorElementRootFolder * subdir, 
			   std::string name);
  /// add contents of folder to private method removedContents
  void add2RemovedContents(const MonitorElementRootFolder * subdir);

  /// -------------------- Unsubscribing/Removing --------------------

  /// remove monitoring element from directory; 
  /// if warning = true, print message if element does not exist
  void removeElement(MonitorElementRootFolder* dir, std::string name,
		     bool warning = true);
  /// remove all monitoring elements from directory; 
  void removeContents(MonitorElementRootFolder * dir);

  /// -------------------- Quality tests on MonitorElements ------------------

   /// add quality report (to be called when test is to run locally)
  QReport * addQReport(MonitorElement * me, QCriterion * qc) const;
  /// add quality report (to be called by ReceiverBase)
  QReport * addQReport(MonitorElement * me, std::string qtname, 
		       QCriterion * qc = 0) const;

  /// run quality tests on all MonitorElements that have been updated (or added)
  /// since last monitoring cycle
  void runQTests(void);

  /// create quality test with unique name <qtname> (analogous to ME name);
  /// quality test can then be attached to ME with useQTest method
  /// (<algo_name> must match one of known algorithms)
  QCriterion * createQTest(std::string algo_name, std::string qtname);

  /// set of all available algorithms for quality tests
  std::set<std::string> availableAlgorithms;

  /// get "global" folder <inpath> status (one of:STATUS_OK, WARNING, ERROR, OTHER);
  /// returns most sever error, where ERROR > WARNING > OTHER > STATUS_OK;
  /// see Core/interface/QTestStatus.h for details on "OTHER" 
  int getStatus(std::string inpath = "") const;
  /// same as above for tag;
  int getStatus(unsigned int tag) const;

  /// same as scanContents in base class but for one path only
  void scanContents(const std::string & search_string, 
		    const MonitorElementRootFolder * folder,
		    std::vector<MonitorElement *> & put_here) const;

  /// run quality tests (also finds updated contents in last monitoring cycle,
  /// including newly added content) <-- to be called only by runQTests
  void runQualityTests(void);
  /// make new directory structure for Subscribers, Tags and CMEsx
  void makeDirStructure(dqm::me_util::rootDir & Dir, std::string name);
  /// read ROOT objects from file <file> in directory <pathname>;
  /// return total # of ROOT objects read
  unsigned int readDirectory(TFile * file, std::string pathname, std::string prepend = "");
  std::string getFileReleaseVersion(std::string filename);
  std::string getFileDQMPatchVersion(std::string filename);

  /// reference histogram (from file) 
  void             readReferenceME(std::string filename); // read from file
  bool             makeReferenceME(MonitorElement* me );  // copy into Referencedir
  bool             isReferenceME(MonitorElement* me) const; // check for existing
  bool             isCollateME(MonitorElement* me) const; // check for existing
  MonitorElement * getReferenceME(MonitorElement * me) const; // refme for given me
  void             deleteME(MonitorElement *me) ; // delete ME (for all me)

  friend class SenderBase;
  friend class ReceiverBase;
  friend class ClientRoot;
  friend class ClientServerRoot;
  friend class MonitorUIRoot;
  friend class DQMTagHelper;
  friend class edm::DQMHttpSource;
};






#endif
