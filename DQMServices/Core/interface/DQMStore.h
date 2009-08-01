#ifndef DQMSERVICES_CORE_DQM_STORE_H
# define DQMSERVICES_CORE_DQM_STORE_H

# include "DQMServices/Core/interface/DQMDefinitions.h"
# include <vector>
# include <string>
# include <list>
# include <map>
# include <set>

namespace edm { class DQMHttpSource; class ParameterSet; }
namespace lat { class Regexp; }

class MonitorElement;
class QCriterion;
class TFile;
class TObject;
class TObjString;
class TH1F;
class TH1S;
class TH2F;
class TH2S;
class TH3F;
class TProfile;
class TProfile2D;

class DQMStore
{
public:
  enum SaveReferenceTag
  {
    SaveWithoutReference,
    SaveWithReference,
    SaveWithReferenceForQTest
  };
  enum OpenRunDirs
  {
    KeepRunDirs,
    StripRunDirs
  };

  //-------------------------------------------------------------------------
  // ---------------------- Constructors ------------------------------------
  DQMStore(const edm::ParameterSet &pset);
  ~DQMStore(void);

  //-------------------------------------------------------------------------
  void				setVerbose(unsigned level);

  // ---------------------- public navigation -------------------------------
  const std::string &		pwd(void) const;
  void				cd(void);
  void				cd(const std::string &subdir);
  void				setCurrentFolder(const std::string &fullpath);
  void				goUp(void);

  bool				dirExists(const std::string &path) const;

  //-------------------------------------------------------------------------
  // ---------------------- public ME booking -------------------------------

  MonitorElement *		bookInt      (const std::string &name);
  MonitorElement *		bookFloat    (const std::string &name);
  MonitorElement *		bookString   (const std::string &name,
					      const std::string &value);
  MonitorElement *		book1D       (const std::string &name,
					      const std::string &title,
					      int nchX, double lowX, double highX);
  MonitorElement *		book1D       (const std::string &name,
					      const std::string &title,
					      int nchX, float *xbinsize);
  MonitorElement *		book1D       (const std::string &name, TH1F *h);

  MonitorElement *		book1S       (const std::string &name,
					      const std::string &title,
					      int nchX, double lowX, double highX);
  MonitorElement *		book1S       (const std::string &name,
					      const std::string &title,
					      int nchX, float *xbinsize);
  MonitorElement *		book1S       (const std::string &name, TH1S *h);

  MonitorElement *		book2D       (const std::string &name,
					      const std::string &title,
					      int nchX, double lowX, double highX,
					      int nchY, double lowY, double highY);
  MonitorElement *		book2D       (const std::string &name,
					      const std::string &title,
					      int nchX, float *xbinsize,
					      int nchY, float *ybinsize);
  MonitorElement *		book2D       (const std::string &name, TH2F *h);

  MonitorElement *		book2S       (const std::string &name,
					      const std::string &title,
					      int nchX, double lowX, double highX,
					      int nchY, double lowY, double highY);
  MonitorElement *		book2S       (const std::string &name,
					      const std::string &title,
					      int nchX, float *xbinsize,
					      int nchY, float *ybinsize);
  MonitorElement *		book2S       (const std::string &name, TH2S *h);

  MonitorElement *		book3D       (const std::string &name,
					      const std::string &title,
					      int nchX, double lowX, double highX,
					      int nchY, double lowY, double highY,
					      int nchZ, double lowZ, double highZ);
  MonitorElement *		book3D       (const std::string &name, TH3F *h);

  MonitorElement *		bookProfile  (const std::string &name,
					      const std::string &title,
					      int nchX, double lowX, double highX,
					      int nchY, double lowY, double highY,
					      const char *option = "s");
  MonitorElement *		bookProfile  (const std::string &name,
					      const std::string &title,
					      int nchX, double lowX, double highX,
					                double lowY, double highY,
					      const char *option = "s");
  MonitorElement *		bookProfile  (const std::string &name, TProfile *h);

  MonitorElement *		bookProfile2D(const std::string &name,
					      const std::string &title,
					      int nchX, double lowX, double highX,
					      int nchY, double lowY, double highY,
					      int nchZ, double lowZ, double highZ,
					      const char *option = "s");
  MonitorElement *		bookProfile2D(const std::string &name,
					      const std::string &title,
					      int nchX, double lowX, double highX,
					      int nchY, double lowY, double highY,
					                double lowZ, double highZ,
					      const char *option = "s");
  MonitorElement *		bookProfile2D(const std::string &name, TProfile2D *h);

  //-------------------------------------------------------------------------
  // ---------------------- public tagging ----------------------------------
  void				tag(MonitorElement *me, unsigned int myTag);
  void				tag(const std::string &path, unsigned int myTag);
  void				tagContents(const std::string &path, unsigned int myTag);
  void				tagAllContents(const std::string &path, unsigned int myTag);

  //-------------------------------------------------------------------------
  // ---------------------- public ME getters -------------------------------
  std::vector<std::string>	getSubdirs(void) const;
  std::vector<std::string>	getMEs(void) const;
  bool				containsAnyMonitorable(const std::string &path) const;

  MonitorElement *		get(const std::string &path) const;
  std::vector<MonitorElement *> get(unsigned int tag) const;
  std::vector<MonitorElement *> getContents(const std::string &path) const;
  std::vector<MonitorElement *> getContents(const std::string &path, unsigned int tag) const;
  void				getContents(std::vector<std::string> &into, bool showContents = true) const;

  // ---------------------- temporarily public for Ecal/Hcal/ -------------
  void				softReset(MonitorElement *me);

  // ---------------------- Public deleting ---------------------------------
  void				rmdir(const std::string &fullpath);
  void				removeContents(void);
  void				removeContents(const std::string &dir);
  void				removeElement(const std::string &name);
  void				removeElement(const std::string &dir, const std::string &name, bool warning = true);

  //-------------------------------------------------------------------------
  // ---------------------- public I/O --------------------------------------
  void				save(const std::string &filename,
				     const std::string &path = "",
				     const std::string &pattern = "",
				     const std::string &rewrite = "",
				     SaveReferenceTag ref = SaveWithReferenceForQTest,
                                     int minStatus = dqm::qstatus::STATUS_OK);
  void				open(const std::string &filename,
				     bool overwrite = false,
				     const std::string &path ="",
				     const std::string &prepend = "");
  void                          load(const std::string &filename,
				     OpenRunDirs stripdirs = StripRunDirs);
  std::string			getFileReleaseVersion(const std::string &filename);
  std::string			getFileDQMPatchVersion(const std::string &filename);
  std::string			getDQMPatchVersion(void);

  //-------------------------------------------------------------------------
  // ---------------------- Public print methods -----------------------------
  void				showDirStructure(void) const;

  //-------------------------------------------------------------------------
  // ---------------------- Quality Test methods -----------------------------
  QCriterion *			getQCriterion(const std::string &qtname) const;
  QCriterion *			createQTest(const std::string &algoname, const std::string &qtname);
  void				useQTest(const std::string &dir, const std::string &qtname);
  int				useQTestByMatch(const std::string &pattern, const std::string &qtname);
  void				runQTests(void);
  int				getStatus(const std::string &path = "") const;

private:
  // ------------ Operations for MEs that are normally never reset ---------
  void				disableSoftReset(MonitorElement *me);

  // ---------------- Navigation -----------------------
  bool				cdInto(const std::string &path) const;

  // ------------------- Reference ME -------------------------------
  bool				makeReferenceME(MonitorElement *me);
  bool				isReferenceME(MonitorElement *me) const;
  bool				isCollateME(MonitorElement *me) const;
  MonitorElement *		getReferenceME(MonitorElement *me) const;

  // ------------------- Private "getters" ------------------------------
  void				readFile(const std::string &filename,
				     bool overwrite = false,
				     const std::string &path ="",
				     const std::string &prepend = "",
				     OpenRunDirs stripdirs = StripRunDirs);
  void				makeDirectory(const std::string &path);
  unsigned int			readDirectory(TFile *file,
					      bool overwrite,
					      const std::string &path,
					      const std::string &prepend,
					      const std::string &curdir,
					      OpenRunDirs stripdirs);

  MonitorElement *		findObject(const std::string &dir,
					   const std::string &name,
					   std::string &path) const;

public:
  void				getAllTags(std::vector<std::string> &into) const;
  std::vector<MonitorElement*>	getAllContents(const std::string &path) const;
  std::vector<MonitorElement*>	getMatchingContents(const std::string &pattern) const;
private:

  // ---------------- Miscellaneous -----------------------------
  void				reset(void);

  bool				extract(TObject *obj, const std::string &dir, bool overwrite);

  // ---------------------- Booking ------------------------------------
  MonitorElement *		initialise(MonitorElement *me, const std::string &path);
  MonitorElement *		book(const std::string &dir, const std::string &name,
				     std::string &path, const char *context);
  template <class HISTO, class COLLATE>
  MonitorElement *		book(const std::string &dir, const std::string &name,
				     const char *context, int kind,
				     HISTO *h, COLLATE collate);

  MonitorElement *		bookInt(const std::string &dir, const std::string &name);
  MonitorElement *		bookFloat(const std::string &dir, const std::string &name);
  MonitorElement *		bookString(const std::string &dir, const std::string &name, const std::string &value);
  MonitorElement *		book1D(const std::string &dir, const std::string &name, TH1F *h);
  MonitorElement *		book1S(const std::string &dir, const std::string &name, TH1S *h);
  MonitorElement *		book2D(const std::string &dir, const std::string &name, TH2F *h);
  MonitorElement *		book2S(const std::string &dir, const std::string &name, TH2S *h);
  MonitorElement *		book3D(const std::string &dir, const std::string &name, TH3F *h);
  MonitorElement *		bookProfile(const std::string &dir, const std::string &name, TProfile *h);
  MonitorElement *		bookProfile2D(const std::string &folder, const std::string &name, TProfile2D *h);

  static void			collate1D(MonitorElement *me, TH1F *h);
  static void			collate1S(MonitorElement *me, TH1S *h);
  static void			collate2D(MonitorElement *me, TH2F *h);
  static void			collate2S(MonitorElement *me, TH2S *h);
  static void			collate3D(MonitorElement *me, TH3F *h);
  static void			collateProfile(MonitorElement *me, TProfile *h);
  static void			collateProfile2D(MonitorElement *me, TProfile2D *h);

  // --- Operations on MEs that are normally reset at end of monitoring cycle ---
  void				setAccumulate(MonitorElement *me, bool flag);

  // ----------------------- singleton admin -----------------------------------
  static DQMStore *		instance(void);

  // ----------------------- Unavailable ---------------------------------------
  DQMStore(const DQMStore&);
  const DQMStore& operator=(const DQMStore&);

  //-------------------------------------------------------------------------------
  //-------------------------------------------------------------------------------
  typedef std::pair<lat::Regexp *, QCriterion *>			QTestSpec;
  typedef std::list<QTestSpec>						QTestSpecs;
  typedef std::map<std::string, MonitorElement>				MEMap;
  typedef std::map<std::string, QCriterion *>				QCMap;
  typedef std::map<std::string, QCriterion *(*)(const std::string &)>	QAMap;
 
  unsigned			verbose_;
  unsigned			verboseQT_;
  bool				reset_;
  bool				collateHistograms_;
  std::string			readSelectedDirectory_;
  bool				outputFileRecreate_;

  std::string			pwd_;
  MEMap				data_;
  std::set<std::string>		dirs_;
  std::vector<std::string>	removed_;

  QCMap				qtests_;
  QAMap				qalgos_;
  QTestSpecs			qtestspecs_;

  friend class edm::DQMHttpSource;
  friend class DQMOldReceiver;
  friend class DQMService;
  friend class DQMNet;
  friend class DQMStoreExample; // for get{All,Matching}Contents -- sole user of this method!
};

#endif // DQMSERVICES_CORE_DQM_STORE_H
