#ifndef DQMSERVICES_CORE_DQM_STORE_H
# define DQMSERVICES_CORE_DQM_STORE_H

# if __GNUC__ && ! defined DQM_DEPRECATED
#  define DQM_DEPRECATED __attribute__((deprecated))
# endif

# include "DQMServices/Core/interface/DQMDefinitions.h"
# include "classlib/utils/Regexp.h"
# include <vector>
# include <string>
# include <list>
# include <map>
# include <set>
# include <cassert>
# include <mutex>
# include <thread>
# include <execinfo.h>
# include <stdio.h>
# include <stdlib.h>
# include <cxxabi.h>

namespace edm { class DQMHttpSource; class ParameterSet; class ActivityRegistry;}
namespace lat { class Regexp; }

class MonitorElement;
class QCriterion;
class TFile;
class TObject;
class TH1;
class TObjString;
class TH1F;
class TH1S;
class TH1D;
class TH2F;
class TH2S;
class TH2D;
class TH3F;
class TProfile;
class TProfile2D;

/** Implements RegEx patterns which occur often in a high-performant
    mattern. For all other expressions, the full RegEx engine is used.
    Note: this class can only be used for lat::Regexp::Wildcard-like
    patterns.  */
class fastmatch
{
 private:
  enum MatchingHeuristicEnum { UseFull, OneStarStart, OneStarEnd, TwoStar };

 public:
  fastmatch (std::string const& _fastString);
  ~fastmatch();

  bool match (std::string const& s) const;

 private:
  // checks if two strings are equal, starting at the back of the strings
  bool compare_strings_reverse (std::string const& pattern,
                                std::string const& input) const;
  // checks if two strings are equal, starting at the front of the strings
  bool compare_strings (std::string const& pattern,
                        std::string const& input) const;

  lat::Regexp * regexp_;
  std::string fastString_;
  MatchingHeuristicEnum matching_;
};

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

  class IBooker
  {
   public:
    friend class DQMStore;

    // for the supported syntaxes, see the declarations of DQMStore::bookString
    template <typename... Args>
    MonitorElement * bookString(Args && ... args) {
      return owner_->bookString(std::forward<Args>(args)...);
    }

    // for the supported syntaxes, see the declarations of DQMStore::bookInt
    template <typename... Args>
    MonitorElement * bookInt(Args && ... args) {
      return owner_->bookInt(std::forward<Args>(args)...);
    }

    // for the supported syntaxes, see the declarations of DQMStore::bookFloat
    template <typename... Args>
    MonitorElement * bookFloat(Args && ... args) {
      return owner_->bookFloat(std::forward<Args>(args)...);
    }

    // for the supported syntaxes, see the declarations of DQMStore::book1D
    template <typename... Args>
    MonitorElement * book1D(Args && ... args) {
      return owner_->book1D(std::forward<Args>(args)...);
    }

    // for the supported syntaxes, see the declarations of DQMStore::book1S
    template <typename... Args>
    MonitorElement * book1S(Args && ... args) {
      return owner_->book1S(std::forward<Args>(args)...);
    }

    // for the supported syntaxes, see the declarations of DQMStore::book1DD
    template <typename... Args>
    MonitorElement * book1DD(Args && ... args) {
      return owner_->book1DD(std::forward<Args>(args)...);
    }

    // for the supported syntaxes, see the declarations of DQMStore::book2D
    template <typename... Args>
    MonitorElement * book2D(Args && ... args) {
      return owner_->book2D(std::forward<Args>(args)...);
    }

    // for the supported syntaxes, see the declarations of DQMStore::book2S
    template <typename... Args>
    MonitorElement * book2S(Args && ... args) {
      return owner_->book2S(std::forward<Args>(args)...);
    }

    // for the supported syntaxes, see the declarations of DQMStore::book2DD
    template <typename... Args>
    MonitorElement * book2DD(Args && ... args) {
      return owner_->book2DD(std::forward<Args>(args)...);
    }

    // for the supported syntaxes, see the declarations of DQMStore::book3D
    template <typename... Args>
    MonitorElement * book3D(Args && ... args) {
      return owner_->book3D(std::forward<Args>(args)...);
    }

    // for the supported syntaxes, see the declarations of DQMStore::bookProfile
    template <typename... Args>
    MonitorElement * bookProfile(Args && ... args) {
      return owner_->bookProfile(std::forward<Args>(args)...);
    }

    // for the supported syntaxes, see the declarations of DQMStore::bookProfile2D
    template <typename... Args>
    MonitorElement * bookProfile2D(Args && ... args) {
      return owner_->bookProfile2D(std::forward<Args>(args)...);
    }

    void cd(void);
    void cd(const std::string &dir);
    void setCurrentFolder(const std::string &fullpath);
    void tag(MonitorElement *, unsigned int);

   private:
    explicit IBooker(DQMStore * store):owner_(0) {
      assert(store);
      owner_ = store;
    }

    IBooker();
    IBooker(const IBooker&);

    // Embedded classes do not natively own a pointer to the embedding
    // class. We therefore need to store a pointer to the main
    // DQMStore instance (owner_).
    DQMStore * owner_;
  };  // IBooker

  class IGetter
  {
   public:
    friend class DQMStore;

    MonitorElement * get(const std::string &path) {
      return owner_->get(path);
    }
    std::vector<std::string> getSubdirs(void) {
      return owner_->getSubdirs();
    }
    std::vector<std::string> getMEs(void) {
      return owner_->getMEs();
    }
    bool containsAnyMonitorable(const std::string &path) {
      return owner_->containsAnyMonitorable(path);
    }
    // for the supported syntaxes, see the declarations of DQMStore::getContents
    template <typename... Args>
    std::vector<MonitorElement *> getContents(Args && ... args) {
      return owner_->getContents(std::forward<Args>(args)...);
    }
    bool dirExists(const std::string &path) {
      return owner_->dirExists(path);
    }

   private:
    explicit IGetter(DQMStore * store):owner_(0) {
      assert(store);
      owner_ = store;
    }

    IGetter();
    IGetter(const IGetter&);

    // Embedded classes do not natively own a pointer to the embedding
    // class. We therefore need to store a pointer to the main
    // DQMStore instance (owner_).
    DQMStore * owner_;
  }; //IGetter

  // Template function to be used inside each DQM Modules' lambda
  // functions to book MonitorElements into the DQMStore. The function
  // calls whatever user-supplied code via the function f. The latter
  // is passed the instance of the IBooker class (owned by the *only*
  // DQMStore instance), that is capable of booking MonitorElements
  // into the DQMStore via a public API. The central mutex is acquired
  // *before* invoking fand automatically released upon returns.
  template <typename iFunc>
  void bookTransaction(iFunc f,
		       uint32_t run,
		       uint32_t streamId,
		       uint32_t moduleId) {
    std::lock_guard<std::mutex> guard(book_mutex_);
    /* If enableMultiThread is not enabled we do not set run_,
       streamId_ and moduleId_ to 0, since we rely on their default
       initialization in DQMSTore constructor. */
    uint32_t oldRun=0,oldStreamId=0,oldModuleId=0;
    if (enableMultiThread_) {
      oldRun = run_;
      run_ = run;
      oldStreamId = streamId_;
      streamId_ = streamId;
      oldModuleId = moduleId_;
      moduleId_ = moduleId;
    }
    f(*ibooker_);
    if (enableMultiThread_) {
      run_ = oldRun;
      streamId_ = oldStreamId;
      moduleId_ = oldModuleId;
    }
  }
  // Signature needed in the harvesting where the booking is done
  // in the endJob. No handles to the run there. Two arguments ensure
  // the capability of booking and getting. The method relies on the
  // initialization of run, stream and module ID to 0. The mutex
  // is not needed.
  template <typename iFunc>
  void meBookerGetter(iFunc f) {
    f(*ibooker_, *igetter_);
  }
  // Signature needed in the harvesting where it might be needed to get
  // the LS based histograms. Handle to the Lumi and to the iSetup are available.
  // No need to book anything there. The method relies on the
  // initialization of run, stream and module ID to 0. The mutex
  // is not needed.
  template <typename iFunc>
  void meGetter(iFunc f) {
    f(*igetter_);
  }

  //-------------------------------------------------------------------------
  // ---------------------- Constructors ------------------------------------
  DQMStore(const edm::ParameterSet &pset, edm::ActivityRegistry&);
  DQMStore(const edm::ParameterSet &pset);
  ~DQMStore(void);

  //-------------------------------------------------------------------------
  void                          setVerbose(unsigned level);

  // ---------------------- public navigation -------------------------------
  const std::string &           pwd(void) const;
  void                          cd(void);
  void                          cd(const std::string &subdir);
  void                          setCurrentFolder(const std::string &fullpath);
  void                          goUp(void);

  bool                          dirExists(const std::string &path) const;

  //-------------------------------------------------------------------------
  // ---------------------- public ME booking -------------------------------

  MonitorElement *              bookInt      (const char *name);
  MonitorElement *              bookInt      (const std::string &name);

  MonitorElement *              bookFloat    (const char *name);
  MonitorElement *              bookFloat    (const std::string &name);

  MonitorElement *              bookString   (const char *name,
                                              const char *value);
  MonitorElement *              bookString   (const std::string &name,
                                              const std::string &value);

  MonitorElement *              book1D       (const char *name,
                                              const char *title,
                                              int nchX, double lowX, double highX);
  MonitorElement *              book1D       (const std::string &name,
                                              const std::string &title,
                                              int nchX, double lowX, double highX);
  MonitorElement *              book1D       (const char *name,
                                              const char *title,
                                              int nchX, float *xbinsize);
  MonitorElement *              book1D       (const std::string &name,
                                              const std::string &title,
                                              int nchX, float *xbinsize);
  MonitorElement *              book1D       (const char *name, TH1F *h);
  MonitorElement *              book1D       (const std::string &name, TH1F *h);

  MonitorElement *              book1S       (const char *name,
                                              const char *title,
                                              int nchX, double lowX, double highX);
  MonitorElement *              book1S       (const std::string &name,
                                              const std::string &title,
                                              int nchX, double lowX, double highX);
  MonitorElement *              book1S       (const char *name,
                                              const char *title,
                                              int nchX, float *xbinsize);
  MonitorElement *              book1S       (const std::string &name,
                                              const std::string &title,
                                              int nchX, float *xbinsize);
  MonitorElement *              book1S       (const char *name, TH1S *h);
  MonitorElement *              book1S       (const std::string &name, TH1S *h);

  MonitorElement *              book1DD       (const char *name,
                                               const char *title,
                                               int nchX, double lowX, double highX);
  MonitorElement *              book1DD       (const std::string &name,
                                               const std::string &title,
                                               int nchX, double lowX, double highX);
  MonitorElement *              book1DD       (const char *name,
                                               const char *title,
                                               int nchX, float *xbinsize);
  MonitorElement *              book1DD       (const std::string &name,
                                               const std::string &title,
                                               int nchX, float *xbinsize);
  MonitorElement *              book1DD       (const char *name, TH1D *h);
  MonitorElement *              book1DD       (const std::string &name, TH1D *h);

  MonitorElement *              book2D       (const char *name,
                                              const char *title,
                                              int nchX, double lowX, double highX,
                                              int nchY, double lowY, double highY);
  MonitorElement *              book2D       (const std::string &name,
                                              const std::string &title,
                                              int nchX, double lowX, double highX,
                                              int nchY, double lowY, double highY);
  MonitorElement *              book2D       (const char *name,
                                              const char *title,
                                              int nchX, float *xbinsize,
                                              int nchY, float *ybinsize);
  MonitorElement *              book2D       (const std::string &name,
                                              const std::string &title,
                                              int nchX, float *xbinsize,
                                              int nchY, float *ybinsize);
  MonitorElement *              book2D       (const char *name, TH2F *h);
  MonitorElement *              book2D       (const std::string &name, TH2F *h);

  MonitorElement *              book2S       (const char *name,
                                              const char *title,
                                              int nchX, double lowX, double highX,
                                              int nchY, double lowY, double highY);
  MonitorElement *              book2S       (const std::string &name,
                                              const std::string &title,
                                              int nchX, double lowX, double highX,
                                              int nchY, double lowY, double highY);
  MonitorElement *              book2S       (const char *name,
                                              const char *title,
                                              int nchX, float *xbinsize,
                                              int nchY, float *ybinsize);
  MonitorElement *              book2S       (const std::string &name,
                                              const std::string &title,
                                              int nchX, float *xbinsize,
                                              int nchY, float *ybinsize);
  MonitorElement *              book2S       (const char *name, TH2S *h);
  MonitorElement *              book2S       (const std::string &name, TH2S *h);

  MonitorElement *              book2DD       (const char *name,
                                               const char *title,
                                               int nchX, double lowX, double highX,
                                               int nchY, double lowY, double highY);
  MonitorElement *              book2DD       (const std::string &name,
                                               const std::string &title,
                                               int nchX, double lowX, double highX,
                                               int nchY, double lowY, double highY);
  MonitorElement *              book2DD       (const char *name,
                                               const char *title,
                                               int nchX, float *xbinsize,
                                               int nchY, float *ybinsize);
  MonitorElement *              book2DD       (const std::string &name,
                                               const std::string &title,
                                               int nchX, float *xbinsize,
                                               int nchY, float *ybinsize);
  MonitorElement *              book2DD       (const char *name, TH2D *h);
  MonitorElement *              book2DD       (const std::string &name, TH2D *h);

  MonitorElement *              book3D       (const char *name,
                                              const char *title,
                                              int nchX, double lowX, double highX,
                                              int nchY, double lowY, double highY,
                                              int nchZ, double lowZ, double highZ);
  MonitorElement *              book3D       (const std::string &name,
                                              const std::string &title,
                                              int nchX, double lowX, double highX,
                                              int nchY, double lowY, double highY,
                                              int nchZ, double lowZ, double highZ);
  MonitorElement *              book3D       (const char *name, TH3F *h);
  MonitorElement *              book3D       (const std::string &name, TH3F *h);

  MonitorElement *              bookProfile  (const char *name,
                                              const char *title,
                                              int nchX, double lowX, double highX,
                                              int nchY, double lowY, double highY,
                                              const char *option = "s");
  MonitorElement *              bookProfile  (const std::string &name,
                                              const std::string &title,
                                              int nchX, double lowX, double highX,
                                              int nchY, double lowY, double highY,
                                              const char *option = "s");
  MonitorElement *              bookProfile  (const char *name,
                                              const char *title,
                                              int nchX, double lowX, double highX,
                                              double lowY, double highY,
                                              const char *option = "s");
  MonitorElement *              bookProfile  (const std::string &name,
                                              const std::string &title,
                                              int nchX, double lowX, double highX,
                                              double lowY, double highY,
                                              const char *option = "s");
  MonitorElement *              bookProfile  (const char *name,
                                              const char *title,
                                              int nchX, double *xbinsize,
                                              int nchY, double lowY, double highY,
                                              const char *option = "s");
  MonitorElement *              bookProfile  (const std::string &name,
                                              const std::string &title,
                                              int nchX, double *xbinsize,
                                              int nchY, double lowY, double highY,
                                              const char *option = "s");
  MonitorElement *              bookProfile  (const char *name,
                                              const char *title,
                                              int nchX, double *xbinsize,
                                              double lowY, double highY,
                                              const char *option = "s");
  MonitorElement *              bookProfile  (const std::string &name,
                                              const std::string &title,
                                              int nchX, double *xbinsize,
                                              double lowY, double highY,
                                              const char *option = "s");
  MonitorElement *              bookProfile  (const char *name, TProfile *h);
  MonitorElement *              bookProfile  (const std::string &name, TProfile *h);

  MonitorElement *              bookProfile2D(const char *name,
                                              const char *title,
                                              int nchX, double lowX, double highX,
                                              int nchY, double lowY, double highY,
                                              int nchZ, double lowZ, double highZ,
                                              const char *option = "s");
  MonitorElement *              bookProfile2D(const std::string &name,
                                              const std::string &title,
                                              int nchX, double lowX, double highX,
                                              int nchY, double lowY, double highY,
                                              int nchZ, double lowZ, double highZ,
                                              const char *option = "s");
  MonitorElement *              bookProfile2D(const char *name,
                                              const char *title,
                                              int nchX, double lowX, double highX,
                                              int nchY, double lowY, double highY,
                                              double lowZ, double highZ,
                                              const char *option = "s");
  MonitorElement *              bookProfile2D(const std::string &name,
                                              const std::string &title,
                                              int nchX, double lowX, double highX,
                                              int nchY, double lowY, double highY,
                                              double lowZ, double highZ,
                                              const char *option = "s");
  MonitorElement *              bookProfile2D(const char *name, TProfile2D *h);
  MonitorElement *              bookProfile2D(const std::string &name, TProfile2D *h);

  //-------------------------------------------------------------------------
  // ---------------------- public tagging ----------------------------------
  void                          tag(MonitorElement *me, unsigned int myTag);
  void                          tag(const std::string &path, unsigned int myTag);
  void                          tagContents(const std::string &path, unsigned int myTag);
  void                          tagAllContents(const std::string &path, unsigned int myTag);

  //-------------------------------------------------------------------------
  // ---------------------- public ME getters -------------------------------
  std::vector<std::string>      getSubdirs(void) const;
  std::vector<std::string>      getMEs(void) const;
  bool                          containsAnyMonitorable(const std::string &path) const;

  MonitorElement *              get(const std::string &path) const;
  std::vector<MonitorElement *> get(unsigned int tag) const;
  std::vector<MonitorElement *> getContents(const std::string &path) const;
  std::vector<MonitorElement *> getContents(const std::string &path, unsigned int tag) const;
  void                          getContents(std::vector<std::string> &into, bool showContents = true) const;

  // ---------------------- softReset methods -------------------------------
  void                          softReset(MonitorElement *me);
  void                          disableSoftReset(MonitorElement *me);

  // ---------------------- Public deleting ---------------------------------
  void                          rmdir(const std::string &fullpath);
  void                          removeContents(void);
  void                          removeContents(const std::string &dir);
  void                          removeElement(const std::string &name);
  void                          removeElement(const std::string &dir, const std::string &name, bool warning = true);

  //-------------------------------------------------------------------------
  // ---------------------- public I/O --------------------------------------
  void                          save(const std::string &filename,
                                     const std::string &path = "",
                                     const std::string &pattern = "",
                                     const std::string &rewrite = "",
                                     const uint32_t run = 0,
                                     SaveReferenceTag ref = SaveWithReference,
                                     int minStatus = dqm::qstatus::STATUS_OK,
                                     const std::string &fileupdate = "RECREATE");
  bool                          open(const std::string &filename,
                                     bool overwrite = false,
                                     const std::string &path ="",
                                     const std::string &prepend = "",
                                     OpenRunDirs stripdirs = KeepRunDirs,
                                     bool fileMustExist = true);
  bool                          load(const std::string &filename,
                                     OpenRunDirs stripdirs = StripRunDirs,
                                     bool fileMustExist = true);

  //-------------------------------------------------------------------------
  // ---------------------- Public print methods -----------------------------
  void                          showDirStructure(void) const;

  // ---------------------- Public check options -----------------------------
  bool                         isCollate(void) const;

  //-------------------------------------------------------------------------
  // ---------------------- Quality Test methods -----------------------------
  QCriterion *                  getQCriterion(const std::string &qtname) const;
  QCriterion *                  createQTest(const std::string &algoname, const std::string &qtname);
  void                          useQTest(const std::string &dir, const std::string &qtname);
  int                           useQTestByMatch(const std::string &pattern, const std::string &qtname);
  void                          runQTests(void);
  int                           getStatus(const std::string &path = "") const;
  void        scaleElements(void);

 private:
  // ---------------- Navigation -----------------------
  bool                          cdInto(const std::string &path) const;

  // ------------------- Reference ME -------------------------------
  bool                          isCollateME(MonitorElement *me) const;

  // ------------------- Private "getters" ------------------------------
  bool                          readFile(const std::string &filename,
                                         bool overwrite = false,
                                         const std::string &path ="",
                                         const std::string &prepend = "",
                                         OpenRunDirs stripdirs = StripRunDirs,
                                         bool fileMustExist = true);
  void                          makeDirectory(const std::string &path);
  unsigned int                  readDirectory(TFile *file,
                                              bool overwrite,
                                              const std::string &path,
                                              const std::string &prepend,
                                              const std::string &curdir,
                                              OpenRunDirs stripdirs);

  MonitorElement *              findObject(const std::string &dir,
                                           const std::string &name,
                                           const uint32_t run = 0,
                                           const uint32_t lumi = 0,
                                           const uint32_t streamId = 0,
                                           const uint32_t moduleId = 0) const;

 public:
  void                          getAllTags(std::vector<std::string> &into) const;
  std::vector<MonitorElement*>  getAllContents(const std::string &path,
                                               uint32_t runNumber = 0,
                                               uint32_t lumi = 0) const;
  std::vector<MonitorElement*>  getMatchingContents(const std::string &pattern, lat::Regexp::Syntax syntaxType = lat::Regexp::Wildcard) const;

  // Multithread SummaryCache manipulations
  void mergeAndResetMEsRunSummaryCache(uint32_t run,
				       uint32_t streamId,
				       uint32_t moduleId);
  void mergeAndResetMEsLuminositySummaryCache(uint32_t run,
					      uint32_t lumi,
					      uint32_t streamId,
					      uint32_t moduleId);
 private:

  // ---------------- Miscellaneous -----------------------------
  void        initializeFrom(const edm::ParameterSet&);
  void                          reset(void);
  void        forceReset(void);

  bool                          extract(TObject *obj, const std::string &dir, bool overwrite);

  // ---------------------- Booking ------------------------------------
  MonitorElement *              initialise(MonitorElement *me, const std::string &path);
  MonitorElement *              book(const std::string &dir,
                                     const std::string &name,
                                     const char *context);
  template <class HISTO, class COLLATE>
  MonitorElement *              book(const std::string &dir, const std::string &name,
                                     const char *context, int kind,
                                     HISTO *h, COLLATE collate);

  MonitorElement *              bookInt(const std::string &dir, const std::string &name);
  MonitorElement *              bookFloat(const std::string &dir, const std::string &name);
  MonitorElement *              bookString(const std::string &dir, const std::string &name, const std::string &value);
  MonitorElement *              book1D(const std::string &dir, const std::string &name, TH1F *h);
  MonitorElement *              book1S(const std::string &dir, const std::string &name, TH1S *h);
  MonitorElement *              book1DD(const std::string &dir, const std::string &name, TH1D *h);
  MonitorElement *              book2D(const std::string &dir, const std::string &name, TH2F *h);
  MonitorElement *              book2S(const std::string &dir, const std::string &name, TH2S *h);
  MonitorElement *              book2DD(const std::string &dir, const std::string &name, TH2D *h);
  MonitorElement *              book3D(const std::string &dir, const std::string &name, TH3F *h);
  MonitorElement *              bookProfile(const std::string &dir, const std::string &name, TProfile *h);
  MonitorElement *              bookProfile2D(const std::string &folder, const std::string &name, TProfile2D *h);

  static bool                   checkBinningMatches(MonitorElement *me, TH1 *h);

  static void                   collate1D(MonitorElement *me, TH1F *h);
  static void                   collate1S(MonitorElement *me, TH1S *h);
  static void                   collate1DD(MonitorElement *me, TH1D *h);
  static void                   collate2D(MonitorElement *me, TH2F *h);
  static void                   collate2S(MonitorElement *me, TH2S *h);
  static void                   collate2DD(MonitorElement *me, TH2D *h);
  static void                   collate3D(MonitorElement *me, TH3F *h);
  static void                   collateProfile(MonitorElement *me, TProfile *h);
  static void                   collateProfile2D(MonitorElement *me, TProfile2D *h);

  // --- Operations on MEs that are normally reset at end of monitoring cycle ---
  void                          setAccumulate(MonitorElement *me, bool flag);

  void print_trace(const std::string &dir, const std::string &name);

  // ----------------------- Unavailable ---------------------------------------
  DQMStore(const DQMStore&);
  const DQMStore& operator=(const DQMStore&);

  //-------------------------------------------------------------------------------
  //-------------------------------------------------------------------------------
  typedef std::pair<fastmatch *, QCriterion *>                  QTestSpec;
  typedef std::list<QTestSpec>                                          QTestSpecs;
  typedef std::set<MonitorElement>                                      MEMap;
  typedef std::map<std::string, QCriterion *>                           QCMap;
  typedef std::map<std::string, QCriterion *(*)(const std::string &)>   QAMap;

  unsigned                      verbose_;
  unsigned                      verboseQT_;
  bool                          reset_;
  double                        scaleFlag_;
  bool                          collateHistograms_;
  bool                          enableMultiThread_;
  std::string                   readSelectedDirectory_;
  uint32_t                      run_;
  uint32_t                      streamId_;
  uint32_t                      moduleId_;

  std::string                   pwd_;
  MEMap                         data_;
  std::set<std::string>         dirs_;

  QCMap                         qtests_;
  QAMap                         qalgos_;
  QTestSpecs                    qtestspecs_;

  std::mutex book_mutex_;
  IBooker * ibooker_;
  IGetter * igetter_;

  friend class edm::DQMHttpSource;
  friend class DQMService;
  friend class DQMNet;
  friend class DQMArchiver;
  friend class DQMStoreExample; // for get{All,Matching}Contents -- sole user of this method!
};

#endif // DQMSERVICES_CORE_DQM_STORE_H

/* Local Variables: */
/* show-trailing-whitespace: t */
/* truncate-lines: t */
/* End: */
