#ifndef DQMServices_Core_DQMStore_h
#define DQMServices_Core_DQMStore_h

#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

#include <type_traits>
#include <functional>
#include <mutex>

// TODO: Remove at some point:
#define TRACE(msg) \
  std::cout << "TRACE: " << __FILE__ << ":" << __LINE__ << "(" << __FUNCTION__ << ") " << msg << std::endl;
#define TRACE_ TRACE("");

namespace dqm {
  namespace implementation {
    using MonitorElement = dqm::legacy::MonitorElement;
    class DQMStore;

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

    class IBooker : public dqm::implementation::NavigatorBase {
    public:
      // functor to be passed as a default argument that does not do anything.
      struct NOOP {
        void operator()(TH1*) const {};
        void operator()() const {};
      };

      //
      // Booking Methods, templated to allow passing in lambdas.
      // The variations taking ROOT object pointers do NOT take ownership of
      // the object; it will be clone'd.
      //

      // The function argument as an optional template parameter adds a lot of
      // ambiguity to the overload resolution, since it accepts *anything* by
      // default (and it does not help that we rely on implicit conversions for
      // almost all of the arguments in many cases, converting string literal
      // to TString and ints to floats, and 0 also prefers to convert to float*
      // and so on ...).
      // So, we use SFINAE to restrict the template parameter type, but that is
      // also not that easy: there is no way to check for sth. callable in
      // type_traits (`is_function` is not the right thing!), so instead we
      // check for not-numeric things, which works most of the time (though e.g.
      // enum constants somehow still pass as not arithmetic and need an
      // explicit cast to resolve the ambiguity).
      template <typename FUNC = NOOP, std::enable_if_t<not std::is_arithmetic<FUNC>::value, int> = 0>
      MonitorElement* bookInt(TString const& name, FUNC onbooking = NOOP()) {
        return bookME(name, MonitorElementData::Kind::INT, [=]() {
          onbooking();
          return nullptr;
        });
      }
      template <typename FUNC = NOOP, std::enable_if_t<not std::is_arithmetic<FUNC>::value, int> = 0>
      MonitorElement* bookFloat(TString const& name, FUNC onbooking = NOOP()) {
        return bookME(name, MonitorElementData::Kind ::REAL, [=]() {
          onbooking();
          return nullptr;
        });
      }
      template <typename FUNC = NOOP, std::enable_if_t<not std::is_arithmetic<FUNC>::value, int> = 0>
      MonitorElement* bookString(TString const& name, TString const& value, FUNC onbooking = NOOP()) {
        // TODO: value unused!
        return bookME(name, MonitorElementData::Kind::STRING, [=]() {
          onbooking();
          return nullptr;
        });
      }

      template <typename FUNC = NOOP, std::enable_if_t<not std::is_arithmetic<FUNC>::value, int> = 0>
      MonitorElement* book1D(TString const& name,
                             TString const& title,
                             int const nchX,
                             double const lowX,
                             double const highX,
                             FUNC onbooking = NOOP()) {
        return bookME(name, MonitorElementData::Kind::TH1F, [=]() {
          auto th1 = new TH1F(name, title, nchX, lowX, highX);
          onbooking(th1);
          return th1;
        });
      }
      template <typename FUNC = NOOP, std::enable_if_t<not std::is_arithmetic<FUNC>::value, int> = 0>
      MonitorElement* book1D(
          TString const& name, TString const& title, int nchX, float const* xbinsize, FUNC onbooking = NOOP()) {
        return bookME(name, MonitorElementData::Kind::TH1F, [=]() {
          auto th1 = new TH1F(name, title, nchX, xbinsize);
          onbooking(th1);
          return th1;
        });
      }
      template <typename FUNC = NOOP, std::enable_if_t<not std::is_arithmetic<FUNC>::value, int> = 0>
      MonitorElement* book1D(TString const& name, TH1F* object, FUNC onbooking = NOOP()) {
        return bookME(name, MonitorElementData::Kind::TH1F, [=]() {
          auto th1 = static_cast<TH1F*>(object->Clone(name));
          onbooking(th1);
          return th1;
        });
      }

      template <typename FUNC = NOOP, std::enable_if_t<not std::is_arithmetic<FUNC>::value, int> = 0>
      MonitorElement* book1S(
          TString const& name, TString const& title, int nchX, double lowX, double highX, FUNC onbooking = NOOP()) {
        return bookME(name, MonitorElementData::Kind::TH1S, [=]() {
          auto th1 = new TH1S(name, title, nchX, lowX, highX);
          onbooking(th1);
          return th1;
        });
      }
      template <typename FUNC = NOOP, std::enable_if_t<not std::is_arithmetic<FUNC>::value, int> = 0>
      MonitorElement* book1S(TString const& name, TH1S* object, FUNC onbooking = NOOP()) {
        return bookME(name, MonitorElementData::Kind::TH1S, [=]() {
          auto th1 = static_cast<TH1S*>(object->Clone(name));
          onbooking(th1);
          return th1;
        });
      }

      template <typename FUNC = NOOP, std::enable_if_t<not std::is_arithmetic<FUNC>::value, int> = 0>
      MonitorElement* book1DD(
          TString const& name, TString const& title, int nchX, double lowX, double highX, FUNC onbooking = NOOP()) {
        return bookME(name, MonitorElementData::Kind::TH1D, [=]() {
          auto th1 = new TH1D(name, title, nchX, lowX, highX);
          onbooking(th1);
          return th1;
        });
      }
      template <typename FUNC = NOOP, std::enable_if_t<not std::is_arithmetic<FUNC>::value, int> = 0>
      MonitorElement* book1DD(TString const& name, TH1D* object, FUNC onbooking = NOOP()) {
        return bookME(name, MonitorElementData::Kind::TH1D, [=]() {
          auto th1 = static_cast<TH1D*>(object->Clone(name));
          onbooking(th1);
          return th1;
        });
      }

      template <typename FUNC = NOOP, std::enable_if_t<not std::is_arithmetic<FUNC>::value, int> = 0>
      MonitorElement* book2D(TString const& name,
                             TString const& title,
                             int nchX,
                             double lowX,
                             double highX,
                             int nchY,
                             double lowY,
                             double highY,
                             FUNC onbooking = NOOP()) {
        return bookME(name, MonitorElementData::Kind::TH2F, [=]() {
          auto th2 = new TH2F(name, title, nchX, lowX, highX, nchY, lowY, highY);
          onbooking(th2);
          return th2;
        });
      }
      template <typename FUNC = NOOP, std::enable_if_t<not std::is_arithmetic<FUNC>::value, int> = 0>
      MonitorElement* book2D(TString const& name,
                             TString const& title,
                             int nchX,
                             float const* xbinsize,
                             int nchY,
                             float const* ybinsize,
                             FUNC onbooking = NOOP()) {
        return bookME(name, MonitorElementData::Kind::TH2F, [=]() {
          auto th2 = new TH2F(name, title, nchX, xbinsize, nchY, ybinsize);
          onbooking(th2);
          return th2;
        });
      }
      template <typename FUNC = NOOP, std::enable_if_t<not std::is_arithmetic<FUNC>::value, int> = 0>
      MonitorElement* book2D(TString const& name, TH2F* object, FUNC onbooking = NOOP()) {
        return bookME(name, MonitorElementData::Kind::TH2F, [=]() {
          auto th2 = static_cast<TH2F*>(object->Clone(name));
          onbooking(th2);
          return th2;
        });
      }
      template <typename FUNC = NOOP, std::enable_if_t<not std::is_arithmetic<FUNC>::value, int> = 0>
      MonitorElement* book2S(TString const& name,
                             TString const& title,
                             int nchX,
                             double lowX,
                             double highX,
                             int nchY,
                             double lowY,
                             double highY,
                             FUNC onbooking = NOOP()) {
        return bookME(name, MonitorElementData::Kind::TH2S, [=]() {
          auto th2 = new TH2S(name, title, nchX, lowX, highX, nchY, lowY, highY);
          onbooking(th2);
          return th2;
        });
      }
      template <typename FUNC = NOOP, std::enable_if_t<not std::is_arithmetic<FUNC>::value, int> = 0>
      MonitorElement* book2S(TString const& name,
                             TString const& title,
                             int nchX,
                             float const* xbinsize,
                             int nchY,
                             float const* ybinsize,
                             FUNC onbooking = NOOP()) {
        return bookME(name, MonitorElementData::Kind::TH2S, [=]() {
          auto th2 = new TH2S(name, title, nchX, xbinsize, nchY, ybinsize);
          onbooking(th2);
          return th2;
        });
      }
      template <typename FUNC = NOOP, std::enable_if_t<not std::is_arithmetic<FUNC>::value, int> = 0>
      MonitorElement* book2S(TString const& name, TH2S* object, FUNC onbooking = NOOP()) {
        return bookME(name, MonitorElementData::Kind::TH2S, [=]() {
          auto th2 = static_cast<TH2S*>(object->Clone(name));
          onbooking(th2);
          return th2;
        });
      }
      template <typename FUNC = NOOP, std::enable_if_t<not std::is_arithmetic<FUNC>::value, int> = 0>
      MonitorElement* book2DD(TString const& name,
                              TString const& title,
                              int nchX,
                              double lowX,
                              double highX,
                              int nchY,
                              double lowY,
                              double highY,
                              FUNC onbooking = NOOP()) {
        return bookME(name, MonitorElementData::Kind::TH2D, [=]() {
          auto th2 = new TH2D(name, title, nchX, lowX, highX, nchY, lowY, highY);
          onbooking(th2);
          return th2;
        });
      }
      template <typename FUNC = NOOP, std::enable_if_t<not std::is_arithmetic<FUNC>::value, int> = 0>
      MonitorElement* book2DD(TString const& name, TH2D* object, FUNC onbooking = NOOP()) {
        return bookME(name, MonitorElementData::Kind::TH2D, [=]() {
          auto th2 = static_cast<TH2D*>(object->Clone(name));
          onbooking(th2);
          return th2;
        });
      }

      template <typename FUNC = NOOP, std::enable_if_t<not std::is_arithmetic<FUNC>::value, int> = 0>
      MonitorElement* book3D(TString const& name,
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
                             FUNC onbooking = NOOP()) {
        return bookME(name, MonitorElementData::Kind::TH3F, [=]() {
          auto th3 = new TH3F(name, title, nchX, lowX, highX, nchY, lowY, highY, nchZ, lowZ, highZ);
          onbooking(th3);
          return th3;
        });
      }
      template <typename FUNC = NOOP, std::enable_if_t<not std::is_arithmetic<FUNC>::value, int> = 0>
      MonitorElement* book3D(TString const& name, TH3F* object, FUNC onbooking = NOOP()) {
        return bookME(name, MonitorElementData::Kind::TH3F, [=]() {
          auto th3 = static_cast<TH3F*>(object->Clone(name));
          onbooking(th3);
          return th3;
        });
      }

      template <typename FUNC = NOOP, std::enable_if_t<not std::is_arithmetic<FUNC>::value, int> = 0>
      MonitorElement* bookProfile(TString const& name,
                                  TString const& title,
                                  int nchX,
                                  double lowX,
                                  double highX,
                                  int /* nchY */,
                                  double lowY,
                                  double highY,
                                  char const* option = "s",
                                  FUNC onbooking = NOOP()) {
        return bookME(name, MonitorElementData::Kind::TPROFILE, [=]() {
          auto tprofile = new TProfile(name, title, nchX, lowX, highX, lowY, highY, option);
          onbooking(tprofile);
          return tprofile;
        });
      }
      template <typename FUNC = NOOP, std::enable_if_t<not std::is_arithmetic<FUNC>::value, int> = 0>
      MonitorElement* bookProfile(TString const& name,
                                  TString const& title,
                                  int nchX,
                                  double lowX,
                                  double highX,
                                  double lowY,
                                  double highY,
                                  char const* option = "s",
                                  FUNC onbooking = NOOP()) {
        return bookME(name, MonitorElementData::Kind::TPROFILE, [=]() {
          auto tprofile = new TProfile(name, title, nchX, lowX, highX, lowY, highY, option);
          onbooking(tprofile);
          return tprofile;
        });
      }
      template <typename FUNC = NOOP, std::enable_if_t<not std::is_arithmetic<FUNC>::value, int> = 0>
      MonitorElement* bookProfile(TString const& name,
                                  TString const& title,
                                  int nchX,
                                  double const* xbinsize,
                                  int /* nchY */,
                                  double lowY,
                                  double highY,
                                  char const* option = "s",
                                  FUNC onbooking = NOOP()) {
        return bookME(name, MonitorElementData::Kind::TPROFILE, [=]() {
          auto tprofile = new TProfile(name, title, nchX, xbinsize, lowY, highY, option);
          onbooking(tprofile);
          return tprofile;
        });
      }
      template <typename FUNC = NOOP, std::enable_if_t<not std::is_arithmetic<FUNC>::value, int> = 0>
      MonitorElement* bookProfile(TString const& name,
                                  TString const& title,
                                  int nchX,
                                  double const* xbinsize,
                                  double lowY,
                                  double highY,
                                  char const* option = "s",
                                  FUNC onbooking = NOOP()) {
        return bookME(name, MonitorElementData::Kind::TPROFILE, [=]() {
          auto tprofile = new TProfile(name, title, nchX, xbinsize, lowY, highY, option);
          onbooking(tprofile);
          return tprofile;
        });
      }
      template <typename FUNC = NOOP, std::enable_if_t<not std::is_arithmetic<FUNC>::value, int> = 0>
      MonitorElement* bookProfile(TString const& name, TProfile* object, FUNC onbooking = NOOP()) {
        return bookME(name, MonitorElementData::Kind::TPROFILE, [=]() {
          auto tprofile = static_cast<TProfile*>(object->Clone(name));
          onbooking(tprofile);
          return tprofile;
        });
      }

      template <typename FUNC = NOOP, std::enable_if_t<not std::is_arithmetic<FUNC>::value, int> = 0>
      MonitorElement* bookProfile2D(TString const& name,
                                    TString const& title,
                                    int nchX,
                                    double lowX,
                                    double highX,
                                    int nchY,
                                    double lowY,
                                    double highY,
                                    double lowZ,
                                    double highZ,
                                    char const* option = "s",
                                    FUNC onbooking = NOOP()) {
        return bookME(name, MonitorElementData::Kind::TPROFILE2D, [=]() {
          auto tprofile = new TProfile2D(name, title, nchX, lowX, highX, nchY, lowY, highY, lowZ, highZ, option);
          onbooking(tprofile);
          return tprofile;
        });
      }
      template <typename FUNC = NOOP, std::enable_if_t<not std::is_arithmetic<FUNC>::value, int> = 0>
      MonitorElement* bookProfile2D(TString const& name,
                                    TString const& title,
                                    int nchX,
                                    double lowX,
                                    double highX,
                                    int nchY,
                                    double lowY,
                                    double highY,
                                    int /* nchZ */,
                                    double lowZ,
                                    double highZ,
                                    char const* option = "s",
                                    FUNC onbooking = NOOP()) {
        return bookME(name, MonitorElementData::Kind::TPROFILE2D, [=]() {
          auto tprofile = new TProfile2D(name, title, nchX, lowX, highX, nchY, lowY, highY, lowZ, highZ, option);
          onbooking(tprofile);
          return tprofile;
        });
      }
      template <typename FUNC = NOOP, std::enable_if_t<not std::is_arithmetic<FUNC>::value, int> = 0>
      MonitorElement* bookProfile2D(TString const& name, TProfile2D* object, FUNC onbooking = NOOP()) {
        return bookME(name, MonitorElementData::Kind::TPROFILE2D, [=]() {
          auto tprofile = static_cast<TProfile2D*>(object->Clone(name));
          onbooking(tprofile);
          return tprofile;
        });
      }

      //
      // all non-template interfaces are virtual.
      //

      virtual MonitorElementData::Scope setScope(MonitorElementData::Scope newscope);
      virtual ~IBooker();

    protected:
      IBooker(DQMStore* store);
      virtual uint64_t setModuleID(uint64_t moduleID);
      virtual edm::LuminosityBlockID setRunLumi(edm::LuminosityBlockID runlumi);
      virtual MonitorElement* bookME(TString const& name,
                                     MonitorElementData::Kind kind,
                                     std::function<TH1*()> makeobject);

      DQMStore* store_ = nullptr;
      MonitorElementData::Scope scope_ = MonitorElementData::Scope::JOB;
      uint64_t moduleID_ = 0;
      edm::LuminosityBlockID runlumi_ = edm::LuminosityBlockID();
    };

    class IGetter : public dqm::implementation::NavigatorBase {
    public:
      // The IGetter interface is not really suitable for concurrent lumis/runs,
      // so we don't even try much to get it right. Harvesting needs to run
      // sequentially for now. Therefore, the methods just return the next-best
      // global MEs that they find, ignoring Scope and run/lumi.
      // Since these are global MEs, they may be deleted at some point; don't
      // store the pointers!
      // They are also shared with other modules. That is save when running
      // multi-threaded as long as getTH1() etc. are not used, but of course
      // all dependencies need to be properly declared to get correct results.

      // TODO: review and possibly rename the all methods below:
      // get MEs that are direct children of full path `path`
      virtual std::vector<dqm::harvesting::MonitorElement*> getContents(std::string const& path) const;
      // not clear what this is good for.
      DQM_DEPRECATED
      virtual void getContents(std::vector<std::string>& into, bool showContents = true) const;

      // get all elements below full path `path`
      // we have to discuss semantics here -- are run/lumi ever used?
      virtual std::vector<dqm::harvesting::MonitorElement*> getAllContents(std::string const& path) const;
      DQM_DEPRECATED
      virtual std::vector<dqm::harvesting::MonitorElement*> getAllContents(std::string const& path,
                                                                           uint32_t runNumber,
                                                                           uint32_t lumi) const;
      // TODO: rename to reflect the fact that it requires full path
      // return ME identified by full path `path`, or nullptr
      virtual MonitorElement* get(std::string const& fullpath) const;

      // same as get, throws an exception if histogram not found
      // Deprecated simply because it is barely used.
      DQM_DEPRECATED
      virtual MonitorElement* getElement(std::string const& path) const;

      // return sub-directories of current directory
      virtual std::vector<std::string> getSubdirs() const;
      // return element names of direct children of current directory
      virtual std::vector<std::string> getMEs() const;
      // returns whether there are objects at full path `path`
      virtual bool dirExists(std::string const& path) const;

      virtual ~IGetter();

    protected:
      IGetter(DQMStore* store);

      DQMStore* store_;
    };

    class DQMStore : public IGetter, public IBooker {
    public:
      // IGetter uses the globalMEs_ directly.
      friend IGetter;
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
      std::vector<MonitorElement*> getMatchingContents(std::string const& pattern) const;

      DQMStore(DQMStore const&) = delete;
      DQMStore& operator=(DQMStore const&) = delete;

      // ------------------------------------------------------------------------
      // ------------ IBooker/IGetter overrides to prevent ambiguity ------------
      virtual void cd() {
        this->IBooker::cd();
        this->IGetter::cd();
      }
      virtual void cd(std::string const& dir) {
        this->IBooker::cd(dir);
        this->IGetter::cd(dir);
      }
      virtual void setCurrentFolder(std::string const& fullpath) {
        this->IBooker::setCurrentFolder(fullpath);
        this->IGetter::setCurrentFolder(fullpath);
      }
      virtual void goUp() {
        this->IBooker::goUp();
        this->IGetter::goUp();
      }
      std::string const& pwd() { return this->IBooker::pwd(); }

    public:
      // internal -- figure out better protection.
      template <typename iFunc>
      void bookTransaction(iFunc f, uint32_t moduleId, bool canSaveByLumi) {
        auto lock = std::scoped_lock(this->booking_mutex_);
        IBooker& booker = *this;
        // TODO: this may need to become more elaborate.
        auto oldscope =
            booker.setScope(canSaveByLumi ? MonitorElementData::Scope::LUMI : MonitorElementData::Scope::RUN);
        assert(moduleId != 0 || !"moduleID must be set for normal booking transaction");
        // Access via this-> to allow access to protected member
        auto oldmoduleid = this->setModuleID(moduleId);
        assert(oldmoduleid == 0 || !"Nested booking transaction?");
        // always book prototypes (except for Scope::JOB, where we can use these directly).
        auto oldrunlumi = this->setRunLumi(edm::LuminosityBlockID());

        f(booker);

        booker.setScope(oldscope);
        this->setModuleID(oldmoduleid);
        this->setRunLumi(oldrunlumi);
      };

      template <typename iFunc>
      void meBookerGetter(iFunc f) {
        auto lock = std::scoped_lock(this->booking_mutex_);
        // here, we make much less assumptions compared to bookTransaction.
        // This is essentially legacy semantics except we actually take the lock.
        f(*this, *this);
        // TODO: we should maybe make sure that Scope changes are reset here,
        // but also it makes sense to inherit the Scope from the environement
        // (e.g. when meBookerGetter is called *inside* a booking transaction).
      };

      // modules are expected to call these callbacks when they change run/lumi.
      // The DQMStore then updates the module's MEs, potentially cloning them
      // if there are concurrent runs/lumis.
      void enterLumi(edm::RunNumber_t run, edm::LuminosityBlockNumber_t lumi, uint64_t moduleID);
      void leaveLumi(edm::RunNumber_t run, edm::LuminosityBlockNumber_t lumi, uint64_t moduleID);

      // Add ME to DQMStore datastructures. The object will be deleted if a
      // similar object is already present.
      // For global ME
      MonitorElement* putME(MonitorElement* me);
      // For local ME
      MonitorElement* putME(MonitorElement* me, uint64_t moduleID);
      // Find a global ME of matching name, in any state.
      // MELIKE can be a MonitorElementData::Path or MonitorElement*.
      template <typename MELIKE>
      MonitorElement* findME(MELIKE const& path);
      // Log a backtrace on booking.
      void printTrace(std::string const& message);

    private:
      // MEComparison is a name-only comparison on MEs and Paths, allowing
      // heterogeneous lookup.
      // The ME objects here are lightweight, all the important stuff is in the
      // MEData. However we never handle MEData directly, but always keep it
      // wrapped in MEs (created as needed). MEs can share MEData.
      // ME objects must never appear more than once in these sets. ME objects
      // in localMEs_ cannot be deleted, since the module might hold pointers.
      // MEs in globalMEs_ can be deleted/recycled at the end of their scope,
      // if there are no MEs left that share the data -- for non-legacy modules
      // that should hold by construction, for legacy MEs (moduleID == 0) it
      // needs an explicit check.
      // MEs can be _protoype MEs_ if their scope is not yet known (after
      // booking, after leaveLumi). A prototype is kept if and only if there is
      // no other global instance of the same ME. Prototype MEs have
      // run = lumi = 0 and scope != JOB. If scope == JOB, a prototype is never
      // required. Prototype MEs are reset *before* inserting, so fill calls
      // can go into prototype MEs and not be lost.
      // Key is (run, lumi), potentially one or both 0 for SCOPE::RUN or SCOPE::JOB
      // NEVER modify the key_ of a ME in these datastructures. Since we use
      // pointers, this may be possible (not everything is const), but it could
      // still corrupt the datastructure.
      std::map<edm::LuminosityBlockID, std::set<MonitorElement*, MonitorElement::MEComparison>> globalMEs_;
      // Key is (moduleID [, run]), run is only needed for edm::global.
      // Legacy MEs have moduleID 0.
      std::map<uint64_t, std::set<MonitorElement*, MonitorElement::MEComparison>> localMEs_;
      // Whenever modifying these sets, take tihs mutex. It's recursive, so we
      // can be liberal -- lock on any access, but also lock on the full booking
      // transaction. The former is required since also the MEComparison is not
      // really thread safe, the latter since booking still uses a single,
      // shared IBooker instance (`this`!), and the transaction needs to be
      // atomic.
      std::recursive_mutex booking_mutex_;

      // universal verbose flag.
      int verbose_;
    };
  }  // namespace implementation

  // Since we still use a single, edm::Serivce instance of a DQMStore, these are all the same.
  namespace legacy {
    class DQMStore : public dqm::implementation::DQMStore {
    public:
      typedef dqm::implementation::IBooker IBooker;
      typedef dqm::implementation::IGetter IGetter;
      using dqm::implementation::DQMStore::DQMStore;
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
