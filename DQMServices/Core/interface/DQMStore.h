#ifndef DQMServices_Core_DQMStore_h
#define DQMServices_Core_DQMStore_h

#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

// TODO: Remove at some point:
#define TRACE(msg) \
  std::cout << "TRACE: " << __FILE__ << ":" << __LINE__ << "(" << __FUNCTION__ << ") " << msg << std::endl;
#define TRACE_ TRACE("");

namespace dqm {
  namespace implementation {
    // The common implementation to change folders
    class NavigatorBase {
    public:
      void cd();
      void cd(std::string const& dir);
      // This is the only method that is allowed to change cwd_ value
      void setCurrentFolder(std::string const& fullpath);
      void goUp();
      std::string const& pwd();

    protected:
      NavigatorBase(){};
      std::string cwd_ = "";
    };
  }

  namespace legacy {

    // The basic IBooker is a virtual interface that returns the common base
    // class of MEs (legacy). That justifies it being in the legacy namespace.
    class IBooker : public dqm::implementation::NavigatorBase {
    public:
      virtual MonitorElement* bookInt(TString const& name) = 0;
      virtual MonitorElement* bookFloat(TString const& name) = 0;
      virtual MonitorElement* bookString(TString const& name, TString const& value) = 0;
      virtual MonitorElement* book1D(
          TString const& name, TString const& title, int const nchX, double const lowX, double const highX) = 0;
      virtual MonitorElement* book1D(TString const& name, TString const& title, int nchX, float const* xbinsize) = 0;
      virtual MonitorElement* book1D(TString const& name, TH1F* object) = 0;
      virtual MonitorElement* book1S(TString const& name, TString const& title, int nchX, double lowX, double highX) = 0;
      virtual MonitorElement* book1S(TString const& name, TH1S* object) = 0;
      virtual MonitorElement* book1DD(
          TString const& name, TString const& title, int nchX, double lowX, double highX) = 0;
      virtual MonitorElement* book1DD(TString const& name, TH1D* object) = 0;
      virtual MonitorElement* book2D(TString const& name,
                                     TString const& title,
                                     int nchX,
                                     double lowX,
                                     double highX,
                                     int nchY,
                                     double lowY,
                                     double highY) = 0;
      virtual MonitorElement* book2D(TString const& name,
                                     TString const& title,
                                     int nchX,
                                     float const* xbinsize,
                                     int nchY,
                                     float const* ybinsize) = 0;
      virtual MonitorElement* book2D(TString const& name, TH2F* object) = 0;
      virtual MonitorElement* book2S(TString const& name,
                                     TString const& title,
                                     int nchX,
                                     double lowX,
                                     double highX,
                                     int nchY,
                                     double lowY,
                                     double highY) = 0;
      virtual MonitorElement* book2S(TString const& name,
                                     TString const& title,
                                     int nchX,
                                     float const* xbinsize,
                                     int nchY,
                                     float const* ybinsize) = 0;
      virtual MonitorElement* book2S(TString const& name, TH2S* object) = 0;
      virtual MonitorElement* book2DD(TString const& name,
                                      TString const& title,
                                      int nchX,
                                      double lowX,
                                      double highX,
                                      int nchY,
                                      double lowY,
                                      double highY) = 0;
      virtual MonitorElement* book2DD(TString const& name, TH2D* object) = 0;
      virtual MonitorElement* book3D(TString const& name,
                                     TString const& title,
                                     int nchX,
                                     double lowX,
                                     double highX,
                                     int nchY,
                                     double lowY,
                                     double highY,
                                     int nchZ,
                                     double lowZ,
                                     double highZ) = 0;
      virtual MonitorElement* book3D(TString const& name, TH3F* object) = 0;
      virtual MonitorElement* bookProfile(TString const& name,
                                          TString const& title,
                                          int nchX,
                                          double lowX,
                                          double highX,
                                          int nchY,
                                          double lowY,
                                          double highY,
                                          char const* option = "s") = 0;
      virtual MonitorElement* bookProfile(TString const& name,
                                          TString const& title,
                                          int nchX,
                                          double lowX,
                                          double highX,
                                          double lowY,
                                          double highY,
                                          char const* option = "s") = 0;
      virtual MonitorElement* bookProfile(TString const& name,
                                          TString const& title,
                                          int nchX,
                                          double const* xbinsize,
                                          int nchY,
                                          double lowY,
                                          double highY,
                                          char const* option = "s") = 0;
      virtual MonitorElement* bookProfile(TString const& name,
                                          TString const& title,
                                          int nchX,
                                          double const* xbinsize,
                                          double lowY,
                                          double highY,
                                          char const* option = "s") = 0;
      virtual MonitorElement* bookProfile(TString const& name, TProfile* object) = 0;
      virtual MonitorElement* bookProfile2D(TString const& name,
                                            TString const& title,
                                            int nchX,
                                            double lowX,
                                            double highX,
                                            int nchY,
                                            double lowY,
                                            double highY,
                                            double lowZ,
                                            double highZ,
                                            char const* option = "s") = 0;
      virtual MonitorElement* bookProfile2D(TString const& name,
                                            TString const& title,
                                            int nchX,
                                            double lowX,
                                            double highX,
                                            int nchY,
                                            double lowY,
                                            double highY,
                                            int nchZ,
                                            double lowZ,
                                            double highZ,
                                            char const* option = "s") = 0;
      virtual MonitorElement* bookProfile2D(TString const& name, TProfile2D* object) = 0;

      virtual MonitorElementData::Scope setScope(MonitorElementData::Scope newscope) = 0;

      virtual ~IBooker();

    protected:
      IBooker();
    };
    class IGetter : public dqm::implementation::NavigatorBase {
    public:
      // TODO: review and possibly rename the all methods below:
      // get MEs that are direct children of full path `path`
      virtual std::vector<dqm::harvesting::MonitorElement*> getContents(std::string const& path) const = 0;
      // not clear what this is good for.
      DQM_DEPRECATED
      virtual void getContents(std::vector<std::string>& into, bool showContents = true) const = 0;

      // get all elements below full path `path`
      // we have to discuss semantics here -- are run/lumi ever used?
      virtual std::vector<dqm::harvesting::MonitorElement*> getAllContents(std::string const& path) const = 0;
      DQM_DEPRECATED
      virtual std::vector<dqm::harvesting::MonitorElement*> getAllContents(std::string const& path,
                                                                           uint32_t runNumber,
                                                                           uint32_t lumi) const = 0;
      // TODO: rename to reflect the fact that it requires full path
      // return ME identified by full path `path`, or nullptr
      virtual MonitorElement* get(std::string const& fullpath) const = 0;

      // same as get, throws an exception if histogram not found
      // Deprecated simply because it is barely used.
      DQM_DEPRECATED
      virtual MonitorElement* getElement(std::string const& path) const = 0;

      // return sub-directories of current directory
      virtual std::vector<std::string> getSubdirs() const = 0;
      // return element names of direct children of current directory
      virtual std::vector<std::string> getMEs() const = 0;
      // returns whether there are objects at full path `path`
      virtual bool dirExists(std::string const& path) const = 0;

      virtual ~IGetter();

    protected:
      IGetter();
    };

  }  // namespace legacy
  // this namespace is for internal use only.
  namespace implementation {
    // this provides a templated implementation of the DQMStore. The operations it
    // does are rather always the same; the only thing that changes are the return
    // types. We keep IBooker and IGetter separate, just in case. DQMStore simply
    // multi-inherits them for now.
    // We will instantiate this for reco MEs and harvesting MEs, and maybe for
    // legacy as well.

    template <class ME, class STORE>
    class IBooker : public dqm::legacy::IBooker {
    public:
      virtual MonitorElementData::Scope setScope(MonitorElementData::Scope newscope);

      virtual ME* bookInt(TString const& name);
      virtual ME* bookFloat(TString const& name);
      virtual ME* bookString(TString const& name, TString const& value);
      virtual ME* book1D(
          TString const& name, TString const& title, int const nchX, double const lowX, double const highX);
      virtual ME* book1D(TString const& name, TString const& title, int nchX, float const* xbinsize);
      virtual ME* book1D(TString const& name, TH1F* object);
      virtual ME* book1S(TString const& name, TString const& title, int nchX, double lowX, double highX);
      virtual ME* book1S(TString const& name, TH1S* object);
      virtual ME* book1DD(TString const& name, TString const& title, int nchX, double lowX, double highX);
      virtual ME* book1DD(TString const& name, TH1D* object);
      virtual ME* book2D(TString const& name,
                         TString const& title,
                         int nchX,
                         double lowX,
                         double highX,
                         int nchY,
                         double lowY,
                         double highY);
      virtual ME* book2D(
          TString const& name, TString const& title, int nchX, float const* xbinsize, int nchY, float const* ybinsize);
      virtual ME* book2D(TString const& name, TH2F* object);
      virtual ME* book2S(TString const& name,
                         TString const& title,
                         int nchX,
                         double lowX,
                         double highX,
                         int nchY,
                         double lowY,
                         double highY);
      virtual ME* book2S(
          TString const& name, TString const& title, int nchX, float const* xbinsize, int nchY, float const* ybinsize);
      virtual ME* book2S(TString const& name, TH2S* object);
      virtual ME* book2DD(TString const& name,
                          TString const& title,
                          int nchX,
                          double lowX,
                          double highX,
                          int nchY,
                          double lowY,
                          double highY);
      virtual ME* book2DD(TString const& name, TH2D* object);
      virtual ME* book3D(TString const& name,
                         TString const& title,
                         int nchX,
                         double lowX,
                         double highX,
                         int nchY,
                         double lowY,
                         double highY,
                         int nchZ,
                         double lowZ,
                         double highZ);
      virtual ME* book3D(TString const& name, TH3F* object);
      virtual ME* bookProfile(TString const& name,
                              TString const& title,
                              int nchX,
                              double lowX,
                              double highX,
                              int nchY,
                              double lowY,
                              double highY,
                              char const* option = "s");
      virtual ME* bookProfile(TString const& name,
                              TString const& title,
                              int nchX,
                              double lowX,
                              double highX,
                              double lowY,
                              double highY,
                              char const* option = "s");
      virtual ME* bookProfile(TString const& name,
                              TString const& title,
                              int nchX,
                              double const* xbinsize,
                              int nchY,
                              double lowY,
                              double highY,
                              char const* option = "s");
      virtual ME* bookProfile(TString const& name,
                              TString const& title,
                              int nchX,
                              double const* xbinsize,
                              double lowY,
                              double highY,
                              char const* option = "s");
      virtual ME* bookProfile(TString const& name, TProfile* object);
      virtual ME* bookProfile2D(TString const& name,
                                TString const& title,
                                int nchX,
                                double lowX,
                                double highX,
                                int nchY,
                                double lowY,
                                double highY,
                                double lowZ,
                                double highZ,
                                char const* option = "s");
      virtual ME* bookProfile2D(TString const& name,
                                TString const& title,
                                int nchX,
                                double lowX,
                                double highX,
                                int nchY,
                                double lowY,
                                double highY,
                                int nchZ,
                                double lowZ,
                                double highZ,
                                char const* option = "s");
      virtual ME* bookProfile2D(TString const& name, TProfile2D* object);

      virtual ~IBooker(){};

    protected:
      IBooker(STORE* store);
      ME* bookME(TString const& name, MonitorElementData::Kind kind, TH1* object);

      STORE* store_;
      MonitorElementData::Scope scope_;
    };

    template <class ME, class STORE>
    class IGetter : public dqm::legacy::IGetter {
    public:
      // TODO: while we can have covariant return types for individual ME*, it seems we can't for the vectors.
      virtual std::vector<dqm::harvesting::MonitorElement*> getContents(std::string const& path) const;
      virtual void getContents(std::vector<std::string>& into, bool showContents = true) const;

      virtual std::vector<dqm::harvesting::MonitorElement*> getAllContents(std::string const& path) const;
      DQM_DEPRECATED
      virtual std::vector<dqm::harvesting::MonitorElement*> getAllContents(std::string const& path,
                                                                           uint32_t runNumber,
                                                                           uint32_t lumi) const;
      virtual ME* get(std::string const& fullpath) const;

      DQM_DEPRECATED
      virtual ME* getElement(std::string const& path) const;

      virtual std::vector<std::string> getSubdirs() const;
      virtual std::vector<std::string> getMEs() const;
      virtual bool dirExists(std::string const& path) const;

      virtual ~IGetter(){};

    protected:
      IGetter(STORE* store);

      STORE* store_;
    };

    template <class ME>
    class DQMStore : public IGetter<ME, DQMStore<ME>>, public IBooker<ME, DQMStore<ME>> {
    public:

      // TODO: There are no references any more. we should gt rid of these.
      enum SaveReferenceTag { SaveWithoutReference, SaveWithReference, SaveWithReferenceForQTest };
      enum OpenRunDirs { KeepRunDirs, StripRunDirs };

      DQMStore(edm::ParameterSet const& pset, edm::ActivityRegistry&);
      ~DQMStore();

      // ------------------------------------------------------------------------
      // ---------------------- public I/O --------------------------------------
      // TODO: these need some cleanup, though in general we want to keep the
      // functionality. Maybe move it to a different class, and change the (rather
      // few) usages.
      void save(std::string const& filename,
                std::string const& path = "",
                std::string const& pattern = "",
                std::string const& rewrite = "",
                uint32_t run = 0,
                uint32_t lumi = 0,
                SaveReferenceTag ref = SaveWithReference,
                int minStatus = dqm::qstatus::STATUS_OK,
                std::string const& fileupdate = "RECREATE");
      void savePB(std::string const& filename, std::string const& path = "", uint32_t run = 0, uint32_t lumi = 0);
      bool open(std::string const& filename,
                bool overwrite = false,
                std::string const& path = "",
                std::string const& prepend = "",
                OpenRunDirs stripdirs = KeepRunDirs,
                bool fileMustExist = true);
      bool load(std::string const& filename, OpenRunDirs stripdirs = StripRunDirs, bool fileMustExist = true);

      DQM_DEPRECATED
      bool mtEnabled() { assert(!"NIY"); }

      DQM_DEPRECATED
      void showDirStructure() const;

      // TODO: getting API not part of IGetter.
      DQM_DEPRECATED
      std::vector<ME*> getMatchingContents(std::string const& pattern) const;

      DQMStore(DQMStore const&) = delete;
      DQMStore& operator=(DQMStore const&) = delete;

      // ------------------------------------------------------------------------
      // ------------ IBooker/IGetter overrides to prevent ambiguity ------------
      virtual void cd() {
        this->IBooker<ME, DQMStore<ME>>::cd();
        this->IGetter<ME, DQMStore<ME>>::cd();
      }
      virtual void cd(std::string const& dir) {
        this->IBooker<ME, DQMStore<ME>>::cd(dir);
        this->IGetter<ME, DQMStore<ME>>::cd(dir);
      }
      virtual void setCurrentFolder(std::string const& fullpath) {
        this->IBooker<ME, DQMStore<ME>>::setCurrentFolder(fullpath);
        this->IGetter<ME, DQMStore<ME>>::setCurrentFolder(fullpath);
      }
      virtual void goUp() {
        this->IBooker<ME, DQMStore<ME>>::goUp();
        this->IGetter<ME, DQMStore<ME>>::goUp();
      }
      std::string const& pwd() { return this->IBooker<ME, DQMStore<ME>>::pwd(); }

    public:
      // internal -- figure out better protection.
      template <typename iFunc>
      void bookTransaction(iFunc f, uint32_t run, uint32_t moduleId, bool canSaveByLumi) {};
      template <typename iFunc>
      void meBookerGetter(iFunc f) {};

      // Make a ME owned by this DQMStore. Will return a pointer to a ME owned
      // by this DQMStore: either an existing ME matching the key of `me` or
      // a newly added one.
      // Will take ownership of the ROOT object in `me`, deleting it if not
      // needed.
      ME* putME(std::unique_ptr<ME>&& me);
      // Log a backtrace on booking.
      void printTrace(std::string const& message);
      // Prepare MEs for the next lumisection. This will create per-lumi copies
      // if ther previous lumi has not yet finished and recycle reusable MEs if
      // booking left any.
      // enterLumi is idempotent; it can be called at any time to update lumi
      // ranges in the MEs. This may be needed in harvesting, where MEs can be
      // booked at any time.
      void enterLumi(edm::RunNumber_t run, edm::LuminosityBlockNumber_t lumi, unsigned int moduleID);
      // Turn the MEs associated with t, run, lumi into a global ME.
      void leaveLumi(edm::RunNumber_t run, edm::LuminosityBlockNumber_t lumi, unsigned int moduleIDi);

      // Clone data including the underlying ROOT object (calls ->Clone()).
      static MonitorElementData* cloneMonitorElementData(MonitorElementData const* input);

      // TODO: Make this section private. localmes_ and inputs_ should be friends with IGetter.
    public:
    };
  }  // namespace implementation

  // Since we still use a single, edm::Serivce instance of a DQMStore, these are all the same.
  namespace legacy {
    class DQMStore : public dqm::implementation::DQMStore<dqm::legacy::MonitorElement> {
    public:
      typedef dqm::legacy::IBooker IBooker;
      typedef dqm::legacy::IGetter IGetter;
      using  dqm::implementation::DQMStore<dqm::legacy::MonitorElement>::DQMStore;
    };
  }  // namespace legacy
  namespace reco {
    typedef dqm::legacy::DQMStore DQMStore;
  }  // namespace reco
  namespace harvesting {
    typedef dqm::legacy::DQMStore DQMStore;
  }  // namespace harvesting
}  // namespace dqm

#endif
