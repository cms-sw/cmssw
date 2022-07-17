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
      virtual void cd();
      // cd is identical to setCurrentFolder!
      DQM_DEPRECATED
      virtual void cd(std::string const& dir);
      // This is the only method that is allowed to change cwd_ value
      virtual void setCurrentFolder(std::string const& fullpath);
      virtual void goUp();
      // returns the current directory without (!) trailing slash or empty string.
      virtual std::string pwd();

      virtual ~NavigatorBase() {}

    protected:
      NavigatorBase(){};
      std::string cwd_ = "";
    };

    class IBooker : public dqm::implementation::NavigatorBase {
    public:
      // DQMStore configures the IBooker in bookTransaction.
      friend class DQMStore;

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
        std::string initial_value(value);
        auto me = bookME(name, MonitorElementData::Kind::STRING, [=]() {
          onbooking();
          return nullptr;
        });
        me->Fill(initial_value);
        return me;
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
        return bookME(
            name,
            MonitorElementData::Kind::TH1F,
            [=]() {
              auto th1 = static_cast<TH1F*>(object->Clone(name));
              onbooking(th1);
              return th1;
            },
            /* forceReplace */ true);
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
        return bookME(
            name,
            MonitorElementData::Kind::TH1S,
            [=]() {
              auto th1 = static_cast<TH1S*>(object->Clone(name));
              onbooking(th1);
              return th1;
            },
            /* forceReplace */ true);
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
        return bookME(
            name,
            MonitorElementData::Kind::TH1D,
            [=]() {
              auto th1 = static_cast<TH1D*>(object->Clone(name));
              onbooking(th1);
              return th1;
            },
            /* forceReplace */ true);
      }

      template <typename FUNC = NOOP, std::enable_if_t<not std::is_arithmetic<FUNC>::value, int> = 0>
      MonitorElement* book1I(TString const& name,
                             TString const& title,
                             int const nchX,
                             double const lowX,
                             double const highX,
                             FUNC onbooking = NOOP()) {
        return bookME(name, MonitorElementData::Kind::TH1I, [=]() {
          auto th1 = new TH1I(name, title, nchX, lowX, highX);
          onbooking(th1);
          return th1;
        });
      }
      template <typename FUNC = NOOP, std::enable_if_t<not std::is_arithmetic<FUNC>::value, int> = 0>
      MonitorElement* book1I(
          TString const& name, TString const& title, int nchX, float const* xbinsize, FUNC onbooking = NOOP()) {
        return bookME(name, MonitorElementData::Kind::TH1I, [=]() {
          auto th1 = new TH1I(name, title, nchX, xbinsize);
          onbooking(th1);
          return th1;
        });
      }
      template <typename FUNC = NOOP, std::enable_if_t<not std::is_arithmetic<FUNC>::value, int> = 0>
      MonitorElement* book1I(TString const& name, TH1I* object, FUNC onbooking = NOOP()) {
        return bookME(
            name,
            MonitorElementData::Kind::TH1I,
            [=]() {
              auto th1 = static_cast<TH1I*>(object->Clone(name));
              onbooking(th1);
              return th1;
            },
            /* forceReplace */ true);
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
        return bookME(
            name,
            MonitorElementData::Kind::TH2F,
            [=]() {
              auto th2 = static_cast<TH2F*>(object->Clone(name));
              onbooking(th2);
              return th2;
            },
            /* forceReplace */ true);
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
        return bookME(
            name,
            MonitorElementData::Kind::TH2S,
            [=]() {
              auto th2 = static_cast<TH2S*>(object->Clone(name));
              onbooking(th2);
              return th2;
            },
            /* forceReplace */ true);
      }
      template <typename FUNC = NOOP, std::enable_if_t<not std::is_arithmetic<FUNC>::value, int> = 0>
      MonitorElement* book2I(TString const& name,
                             TString const& title,
                             int nchX,
                             double lowX,
                             double highX,
                             int nchY,
                             double lowY,
                             double highY,
                             FUNC onbooking = NOOP()) {
        return bookME(name, MonitorElementData::Kind::TH2I, [=]() {
          auto th2 = new TH2I(name, title, nchX, lowX, highX, nchY, lowY, highY);
          onbooking(th2);
          return th2;
        });
      }
      template <typename FUNC = NOOP, std::enable_if_t<not std::is_arithmetic<FUNC>::value, int> = 0>
      MonitorElement* book2I(TString const& name,
                             TString const& title,
                             int nchX,
                             float const* xbinsize,
                             int nchY,
                             float const* ybinsize,
                             FUNC onbooking = NOOP()) {
        return bookME(name, MonitorElementData::Kind::TH2I, [=]() {
          auto th2 = new TH2I(name, title, nchX, xbinsize, nchY, ybinsize);
          onbooking(th2);
          return th2;
        });
      }
      template <typename FUNC = NOOP, std::enable_if_t<not std::is_arithmetic<FUNC>::value, int> = 0>
      MonitorElement* book2I(TString const& name, TH2I* object, FUNC onbooking = NOOP()) {
        return bookME(
            name,
            MonitorElementData::Kind::TH2I,
            [=]() {
              auto th2 = static_cast<TH2I*>(object->Clone(name));
              onbooking(th2);
              return th2;
            },
            /* forceReplace */ true);
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
        return bookME(
            name,
            MonitorElementData::Kind::TH2D,
            [=]() {
              auto th2 = static_cast<TH2D*>(object->Clone(name));
              onbooking(th2);
              return th2;
            },
            /* forceReplace */ true);
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
        return bookME(
            name,
            MonitorElementData::Kind::TH3F,
            [=]() {
              auto th3 = static_cast<TH3F*>(object->Clone(name));
              onbooking(th3);
              return th3;
            },
            /* forceReplace */ true);
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
        return bookME(
            name,
            MonitorElementData::Kind::TPROFILE,
            [=]() {
              auto tprofile = static_cast<TProfile*>(object->Clone(name));
              onbooking(tprofile);
              return tprofile;
            },
            /* forceReplace */ true);
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
        return bookME(
            name,
            MonitorElementData::Kind::TPROFILE2D,
            [=]() {
              auto tprofile = static_cast<TProfile2D*>(object->Clone(name));
              onbooking(tprofile);
              return tprofile;
            },
            /* forceReplace */ true);
      }

      //
      // all non-template interfaces are virtual.
      //

      virtual MonitorElementData::Scope setScope(MonitorElementData::Scope newscope);
      // RAII-Style guard to set and reset the Scope.
      template <MonitorElementData::Scope SCOPE>
      struct UseScope {
        IBooker& parent;
        MonitorElementData::Scope oldscope;
        UseScope(IBooker& booker) : parent(booker) { oldscope = parent.setScope(SCOPE); }
        ~UseScope() { parent.setScope(oldscope); }
      };
      using UseLumiScope = UseScope<MonitorElementData::Scope::LUMI>;
      using UseRunScope = UseScope<MonitorElementData::Scope::RUN>;
      using UseJobScope = UseScope<MonitorElementData::Scope::JOB>;

      ~IBooker() override;

    private:
      IBooker(DQMStore* store);
      virtual uint64_t setModuleID(uint64_t moduleID);
      virtual edm::LuminosityBlockID setRunLumi(edm::LuminosityBlockID runlumi);
      virtual MonitorElement* bookME(TString const& name,
                                     MonitorElementData::Kind kind,
                                     std::function<TH1*()> makeobject,
                                     bool forceReplace = false);

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

      // This is the most specific way to get a ME, specifying also run and
      // lumi in the key. Primarily for internal use.
      virtual MonitorElement* get(MonitorElementData::Key const& key) const;

      // same as get, throws an exception if histogram not found
      // Deprecated simply because it is barely used.
      DQM_DEPRECATED
      virtual MonitorElement* getElement(std::string const& path) const;

      // return sub-directories of current directory
      // Deprecated because the current implementation is very slow and barely
      // used, use getAllContents instead.
      DQM_DEPRECATED
      virtual std::vector<std::string> getSubdirs() const;
      // return element names of direct children of current directory
      virtual std::vector<std::string> getMEs() const;
      // returns whether there are objects at full path `path`
      virtual bool dirExists(std::string const& path) const;

      ~IGetter() override;

    protected:
      IGetter(DQMStore* store);

      DQMStore* store_;
    };

    class DQMStore : public IGetter, public IBooker {
    public:
      // IGetter uses the globalMEs_ directly.
      friend IGetter;
      enum OpenRunDirs { KeepRunDirs, StripRunDirs };

      DQMStore(edm::ParameterSet const& pset, edm::ActivityRegistry&);
      DQMStore(DQMStore const&) = delete;
      DQMStore(DQMStore&&) = delete;
      DQMStore& operator=(DQMStore const&) = delete;
      ~DQMStore() override;

      // ------------------------------------------------------------------------
      // ---------------------- public I/O --------------------------------------
      DQM_DEPRECATED
      void save(std::string const& filename, std::string const& path = "");
      DQM_DEPRECATED
      bool open(std::string const& filename,
                bool overwrite = false,
                std::string const& path = "",
                std::string const& prepend = "",
                OpenRunDirs stripdirs = KeepRunDirs,
                bool fileMustExist = true);

      // ------------------------------------------------------------------------
      // ------------ IBooker/IGetter overrides to prevent ambiguity ------------
      void cd() override { this->IBooker::cd(); }
      void cd(std::string const& dir) override { this->IBooker::cd(dir); }
      void goUp() override { this->IBooker::goUp(); }
      std::string pwd() override { return this->IBooker::pwd(); }

      void setCurrentFolder(std::string const& fullpath) override {
        // only here we keep the two in sync -- all the others call this in the end!
        this->IBooker::setCurrentFolder(fullpath);
        this->IGetter::setCurrentFolder(fullpath);
      }

    public:
      // internal -- figure out better protection.
      template <typename iFunc>
      void bookTransaction(iFunc f, uint64_t moduleId, bool canSaveByLumi) {
        // taking the lock here only to protect the single, global IBooker (as
        // base class of DQMStore). We could avoid taking this lock by making a
        // new IBooker instance for each transaction, and the DQMStore itself
        // takes the lock before touching any data structures.
        // There is a race in bookME when we don't take this lock, where two
        // modules might prepare a global ME for the same name at the same time
        // and only one of them succeeds in putME: this is is safe, but we need
        // to remove the assertion over there and subsystem code has to be
        // aware that the booking callback *can* run multiple times.
        // Additionally, this lock is what keeps usage of getTH1() safe during
        // booking... all code needs to be migrated to callbacks before this can
        // be removed.
        auto lock = std::scoped_lock(this->booking_mutex_);

        // This is to make sure everything gets reset in case of an exception.
        // That is not really required (an exception here will crash the job
        // anyways, and it is technically not required to reset everything), but
        // it prevents misleading error messages in other threads.
        struct ModuleIdScope {
          IBooker& booker_;
          uint64_t oldid_;
          MonitorElementData::Scope oldscope_;
          edm::LuminosityBlockID oldrunlumi_;
          ModuleIdScope(IBooker& booker,
                        uint64_t newid,
                        MonitorElementData::Scope newscope,
                        edm::LuminosityBlockID newrunlumi)
              : booker_(booker) {
            oldid_ = booker_.setModuleID(newid);
            oldscope_ = booker_.setScope(newscope);
            oldrunlumi_ = booker_.setRunLumi(newrunlumi);
            assert(newid != 0 || !"moduleID must be set for normal booking transaction");
            assert(oldid_ == 0 || !"Nested booking transaction?");
          }
          ~ModuleIdScope() {
            booker_.setModuleID(oldid_);
            booker_.setScope(oldscope_);
            booker_.setRunLumi(oldrunlumi_);
          }
        };

        ModuleIdScope booker(
            *this,
            moduleId,
            // enable per-lumi-by-default here
            canSaveByLumi && this->doSaveByLumi_ ? MonitorElementData::Scope::LUMI : MonitorElementData::Scope::RUN,
            // always book prototypes
            edm::LuminosityBlockID());

        f(booker.booker_);
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

      // For input modules: trigger recycling without local ME/enterLumi/moduleID.
      MonitorElement* findOrRecycle(MonitorElementData::Key const&);

      // this creates local all needed global MEs for the given run/lumi (and
      // module), potentially cloning them if there are concurrent runs/lumis.
      // Symmetrical to cleanupLumi, this is called from a framwork hook, to
      // make sure it also runs when the module does not call anything.
      void initLumi(edm::RunNumber_t run, edm::LuminosityBlockNumber_t lumi);
      void initLumi(edm::RunNumber_t run, edm::LuminosityBlockNumber_t lumi, uint64_t moduleID);

      // modules are expected to call these callbacks when they change run/lumi.
      // The DQMStore then updates the module's MEs local MEs to point to the
      // new run/lumi.
      void enterLumi(edm::RunNumber_t run, edm::LuminosityBlockNumber_t lumi, uint64_t moduleID);
      void leaveLumi(edm::RunNumber_t run, edm::LuminosityBlockNumber_t lumi, uint64_t moduleID);

      // this is triggered by a framework hook to remove/recycle MEs after a
      // run/lumi is saved. We do this via the edm::Service interface to make
      // sure it runs after *all* output modules, even if there are multiple.
      void cleanupLumi(edm::RunNumber_t run, edm::LuminosityBlockNumber_t lumi);

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
      // print a log message if ME matches trackME_.
      void debugTrackME(const char* message, MonitorElement* me_local, MonitorElement* me_global) const;

    private:
      // MEComparison is a name-only comparison on MEs and Paths, allowing
      // heterogeneous lookup.
      // The ME objects here are lightweight, all the important stuff is in the
      // MEData. However we never handle MEData directly, but always keep it
      // wrapped in MEs (created as needed). MEs can share MEData.
      // ME objects must never appear more than once in these sets. ME objects
      // in localMEs_ cannot be deleted, since the module might hold pointers.
      // MEs in globalMEs_ can be deleted/recycled at the end of their scope,
      // if there are no local MEs left that share the data.
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
      // Key is (moduleID [, run | , stream]), run is only needed for
      // edm::global, stream only for edm::stream.
      // Legacy MEs have moduleID 0.
      std::map<uint64_t, std::set<MonitorElement*, MonitorElement::MEComparison>> localMEs_;
      // Whenever modifying these sets, take this  mutex. It's recursive, so we
      // can be liberal -- lock on any access, but also lock on the full booking
      // transaction.
      std::recursive_mutex booking_mutex_;

      // Universal verbose flag.
      // Only very few usages remain, the main debugging tool is trackME_.
      int verbose_;

      // If set to true, error out whenever things happen that are not safe for
      // legacy modules.
      bool assertLegacySafe_;

      // Book MEs by lumi by default whenever possible.
      bool doSaveByLumi_;
      std::vector<std::string> MEsToSave_;  //just if perLS is ON

      // if non-empty, debugTrackME calls will log some information whenever a
      // ME path contains this string.
      std::string trackME_;
    };
  }  // namespace implementation

  // Since we still use a single, edm::Serivce instance of a DQMStore, these are all the same.
  namespace legacy {
    class DQMStore : public dqm::implementation::DQMStore {
    public:
      using IBooker = dqm::implementation::IBooker;
      using IGetter = dqm::implementation::IGetter;
      // import constructors.
      using dqm::implementation::DQMStore::DQMStore;
    };
  }  // namespace legacy
  namespace reco {
    using DQMStore = dqm::legacy::DQMStore;
  }  // namespace reco
  namespace harvesting {
    using DQMStore = dqm::legacy::DQMStore;
  }  // namespace harvesting
}  // namespace dqm

#endif
