// silence deprecation warnings for the DQMStore itself.
#define DQM_DEPRECATED
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <string>
#include <regex>
#include <csignal>

#include <execinfo.h>
#include <cxxabi.h>

namespace dqm {

  namespace legacy {
    IBooker::IBooker() {}
    IBooker::~IBooker() {}
    IGetter::IGetter() {}
    IGetter::~IGetter() {}

  }  // namespace legacy

  namespace implementation {
    std::string const& NavigatorBase::pwd() { return cwd_; }
    void NavigatorBase::cd() { setCurrentFolder(""); }
    void NavigatorBase::cd(std::string const& dir) { setCurrentFolder(cwd_ + dir); }
    void NavigatorBase::goUp() { cd(".."); }
    void NavigatorBase::setCurrentFolder(std::string const& fullpath) {
      MonitorElementData::Path path;
      path.set(fullpath, MonitorElementData::Path::Type::DIR);
      assert(this);
      cwd_ = path.getDirname();
    }

    template <class ME, class STORE>
    IBooker<ME, STORE>::IBooker(STORE* store) {
      store_ = store;
      scope_ = MonitorElementData::Scope::RUN;
    }

    template <class ME, class STORE>
    MonitorElementData::Scope IBooker<ME, STORE>::setScope(MonitorElementData::Scope newscope) {
      auto oldscope = scope_;
      scope_ = newscope;
      return oldscope;
    }

    template <class ME, class STORE>
    ME* IBooker<ME, STORE>::bookME(TString const& name, MonitorElementData::Kind kind, TH1* object) {
      MonitorElementData* data = new MonitorElementData();
      MonitorElementData::Key key;
      key.kind_ = kind;
      std::string fullpath = pwd() + std::string(name.View());
      key.path_.set(fullpath, MonitorElementData::Path::Type::DIR_AND_NAME);
      key.scope_ = scope_;
      data->key_ = key;
      {
        //MonitorElementData::Value::Access value(data->value_);
        //value.object = std::unique_ptr<TH1>(object);
      }

      std::unique_ptr<ME> me = std::make_unique<ME>(data, /* is_owned */ true, /* is_readonly */ false);
      assert(me);
      ME* me_ptr = store_->putME(std::move(me));
      assert(me_ptr);
      return me_ptr;
    }

    template <class ME>
    ME* DQMStore<ME>::putME(std::unique_ptr<ME>&& me) {
      //TODO
      return nullptr;
    }

    template <class ME>
    void DQMStore<ME>::printTrace(std::string const& message) {
      edm::LogWarning("DQMStoreBooking").log([&](auto& logger) {
        std::regex s_rxtrace{"(.*)\\((.*)\\+0x.*\\).*(\\[.*\\])"};
        std::regex s_rxself{"^[^()]*dqm::implementation::.*|^[^()]*edm::.*|.*edm::convertException::wrap.*"};

        void* array[10];
        size_t size;
        char** strings;
        int demangle_status = 0;
        std::vector<std::string> clean_trace;

        // glibc/libgcc backtrace functionality, declared in execinfo.h.
        size = backtrace(array, 10);
        strings = backtrace_symbols(array, size);

        size_t level = 1;
        char* demangled = nullptr;
        for (; level < size; ++level) {
          std::cmatch match;
          bool ok = std::regex_match(strings[level], match, s_rxtrace);

          if (!ok) {
            edm::LogWarning("DQMStoreBacktrace") << "failed match" << level << strings[level];
            continue;
          }

          if (match[2].length() == 0) {
            // no symbol, ignore.
            continue;
          }

          // demangle name to human readable form
          demangled = abi::__cxa_demangle(std::string(match[2]).c_str(), nullptr, nullptr, &demangle_status);
          if (!demangled || demangle_status != 0) {
            edm::LogWarning("DQMStoreBacktrace") << "failed demangle! status " << demangle_status << " on " << match[2];
            continue;
          }

          if (std::regex_match(demangled, s_rxself)) {
            // ignore framework/internal methods
            free(demangled);
            demangled = nullptr;
            continue;
          } else {
            // keep the demangled name and the address.
            // The address can be resolved to a line number in gdb attached to
            // the process, using `list *0x<addr>`, but it can only be done in
            // the running process and we can"t easily do it in this code.
            clean_trace.push_back(std::string(demangled) + std::string(match[3]));
            free(demangled);
            demangled = nullptr;
          }
        }

        if (clean_trace.size() > 0) {
          logger << message << " at ";
          for (auto const& s : clean_trace) {
            logger << s << "; ";
          }
        } else {
          logger << message << " : failed to collect stack trace.";
        }

        free(strings);
      });
    }

    template <class ME, class STORE>
    ME* IBooker<ME, STORE>::bookInt(TString const& name) {
      return bookME(name, MonitorElementData::Kind::INT, nullptr);
    }
    template <class ME, class STORE>
    ME* IBooker<ME, STORE>::bookFloat(TString const& name) {
      return bookME(name, MonitorElementData::Kind::INT, nullptr);
    }
    template <class ME, class STORE>
    ME* IBooker<ME, STORE>::bookString(TString const& name, TString const& value) {
      return bookME(name, MonitorElementData::Kind::INT, nullptr);
    }
    template <class ME, class STORE>
    ME* IBooker<ME, STORE>::book1D(
        TString const& name, TString const& title, int const nchX, double const lowX, double const highX) {
      auto th1 = new TH1F(name, title, nchX, lowX, highX);
      return bookME(name, MonitorElementData::Kind::TH1F, th1);
    }
    template <class ME, class STORE>
    ME* IBooker<ME, STORE>::book1D(TString const& name, TString const& title, int nchX, float const* xbinsize) {
      auto th1 = new TH1F(name, title, nchX, xbinsize);
      return bookME(name, MonitorElementData::Kind::TH1F, th1);
    }
    template <class ME, class STORE>
    ME* IBooker<ME, STORE>::book1D(TString const& name, TH1F* object) {
      auto th1 = static_cast<TH1*>(object->Clone(name));
      return bookME(name, MonitorElementData::Kind::TH1F, th1);
    }
    template <class ME, class STORE>
    ME* IBooker<ME, STORE>::book1S(TString const& name, TString const& title, int nchX, double lowX, double highX) {
      auto th1 = new TH1S(name, title, nchX, lowX, highX);
      return bookME(name, MonitorElementData::Kind::TH1S, th1);
    }
    template <class ME, class STORE>
    ME* IBooker<ME, STORE>::book1S(TString const& name, TH1S* object) {
      auto th1 = static_cast<TH1*>(object->Clone(name));
      return bookME(name, MonitorElementData::Kind::TH1S, th1);
    }
    template <class ME, class STORE>
    ME* IBooker<ME, STORE>::book1DD(TString const& name, TString const& title, int nchX, double lowX, double highX) {
      auto th1 = new TH1D(name, title, nchX, lowX, highX);
      return bookME(name, MonitorElementData::Kind::TH1D, th1);
    }
    template <class ME, class STORE>
    ME* IBooker<ME, STORE>::book1DD(TString const& name, TH1D* object) {
      auto th1 = static_cast<TH1*>(object->Clone(name));
      return bookME(name, MonitorElementData::Kind::TH1D, th1);
    }
    template <class ME, class STORE>
    ME* IBooker<ME, STORE>::book2D(TString const& name,
                                   TString const& title,
                                   int nchX,
                                   double lowX,
                                   double highX,
                                   int nchY,
                                   double lowY,
                                   double highY) {
      auto th2 = new TH2F(name, title, nchX, lowX, highX, nchY, lowY, highY);
      return bookME(name, MonitorElementData::Kind::TH2F, th2);
    }
    template <class ME, class STORE>
    ME* IBooker<ME, STORE>::book2D(
        TString const& name, TString const& title, int nchX, float const* xbinsize, int nchY, float const* ybinsize) {
      auto th2 = new TH2F(name, title, nchX, xbinsize, nchY, ybinsize);
      return bookME(name, MonitorElementData::Kind::TH2F, th2);
    }
    template <class ME, class STORE>
    ME* IBooker<ME, STORE>::book2D(TString const& name, TH2F* object) {
      auto th2 = static_cast<TH1*>(object->Clone(name));
      return bookME(name, MonitorElementData::Kind::TH2F, th2);
    }
    template <class ME, class STORE>
    ME* IBooker<ME, STORE>::book2S(TString const& name,
                                   TString const& title,
                                   int nchX,
                                   double lowX,
                                   double highX,
                                   int nchY,
                                   double lowY,
                                   double highY) {
      auto th2 = new TH2S(name, title, nchX, lowX, highX, nchY, lowY, highY);
      return bookME(name, MonitorElementData::Kind::TH2S, th2);
    }
    template <class ME, class STORE>
    ME* IBooker<ME, STORE>::book2S(
        TString const& name, TString const& title, int nchX, float const* xbinsize, int nchY, float const* ybinsize) {
      auto th2 = new TH2S(name, title, nchX, xbinsize, nchY, ybinsize);
      return bookME(name, MonitorElementData::Kind::TH2S, th2);
    }
    template <class ME, class STORE>
    ME* IBooker<ME, STORE>::book2S(TString const& name, TH2S* object) {
      auto th2 = static_cast<TH1*>(object->Clone(name));
      return bookME(name, MonitorElementData::Kind::TH2S, th2);
    }
    template <class ME, class STORE>
    ME* IBooker<ME, STORE>::book2DD(TString const& name,
                                    TString const& title,
                                    int nchX,
                                    double lowX,
                                    double highX,
                                    int nchY,
                                    double lowY,
                                    double highY) {
      auto th2 = new TH2D(name, title, nchX, lowX, highX, nchY, lowY, highY);
      return bookME(name, MonitorElementData::Kind::TH2D, th2);
    }
    template <class ME, class STORE>
    ME* IBooker<ME, STORE>::book2DD(TString const& name, TH2D* object) {
      auto th2 = static_cast<TH1*>(object->Clone(name));
      return bookME(name, MonitorElementData::Kind::TH2D, th2);
    }
    template <class ME, class STORE>
    ME* IBooker<ME, STORE>::book3D(TString const& name,
                                   TString const& title,
                                   int nchX,
                                   double lowX,
                                   double highX,
                                   int nchY,
                                   double lowY,
                                   double highY,
                                   int nchZ,
                                   double lowZ,
                                   double highZ) {
      auto th3 = new TH3F(name, title, nchX, lowX, highX, nchY, lowY, highY, nchZ, lowZ, highZ);
      return bookME(name, MonitorElementData::Kind::TH3F, th3);
    }
    template <class ME, class STORE>
    ME* IBooker<ME, STORE>::book3D(TString const& name, TH3F* object) {
      auto th3 = static_cast<TH1*>(object->Clone(name));
      return bookME(name, MonitorElementData::Kind::TH3F, th3);
    }
    template <class ME, class STORE>
    ME* IBooker<ME, STORE>::bookProfile(TString const& name,
                                        TString const& title,
                                        int nchX,
                                        double lowX,
                                        double highX,
                                        int /* nchY */,
                                        double lowY,
                                        double highY,
                                        char const* option) {
      auto tprofile = new TProfile(name, title, nchX, lowX, highX, lowY, highY, option);
      return bookME(name, MonitorElementData::Kind::TPROFILE, tprofile);
    }
    template <class ME, class STORE>
    ME* IBooker<ME, STORE>::bookProfile(TString const& name,
                                        TString const& title,
                                        int nchX,
                                        double lowX,
                                        double highX,
                                        double lowY,
                                        double highY,
                                        char const* option) {
      auto tprofile = new TProfile(name, title, nchX, lowX, highX, lowY, highY, option);
      return bookME(name, MonitorElementData::Kind::TPROFILE, tprofile);
    }
    template <class ME, class STORE>
    ME* IBooker<ME, STORE>::bookProfile(TString const& name,
                                        TString const& title,
                                        int nchX,
                                        double const* xbinsize,
                                        int /* nchY */,
                                        double lowY,
                                        double highY,
                                        char const* option) {
      auto tprofile = new TProfile(name, title, nchX, xbinsize, lowY, highY, option);
      return bookME(name, MonitorElementData::Kind::TPROFILE, tprofile);
    }
    template <class ME, class STORE>
    ME* IBooker<ME, STORE>::bookProfile(TString const& name,
                                        TString const& title,
                                        int nchX,
                                        double const* xbinsize,
                                        double lowY,
                                        double highY,
                                        char const* option) {
      auto tprofile = new TProfile(name, title, nchX, xbinsize, lowY, highY, option);
      return bookME(name, MonitorElementData::Kind::TPROFILE, tprofile);
    }
    template <class ME, class STORE>
    ME* IBooker<ME, STORE>::bookProfile(TString const& name, TProfile* object) {
      auto tprofile = static_cast<TH1*>(object->Clone(name));
      return bookME(name, MonitorElementData::Kind::TPROFILE, tprofile);
    }
    template <class ME, class STORE>
    ME* IBooker<ME, STORE>::bookProfile2D(TString const& name,
                                          TString const& title,
                                          int nchX,
                                          double lowX,
                                          double highX,
                                          int nchY,
                                          double lowY,
                                          double highY,
                                          double lowZ,
                                          double highZ,
                                          char const* option) {
      auto tprofile = new TProfile2D(name, title, nchX, lowX, highX, nchY, lowY, highY, lowZ, highZ, option);
      return bookME(name, MonitorElementData::Kind::TPROFILE2D, tprofile);
    }
    template <class ME, class STORE>
    ME* IBooker<ME, STORE>::bookProfile2D(TString const& name,
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
                                          char const* option) {
      auto tprofile = new TProfile2D(name, title, nchX, lowX, highX, nchY, lowY, highY, lowZ, highZ, option);
      return bookME(name, MonitorElementData::Kind::TPROFILE2D, tprofile);
    }
    template <class ME, class STORE>
    ME* IBooker<ME, STORE>::bookProfile2D(TString const& name, TProfile2D* object) {
      auto tprofile = static_cast<TH1*>(object->Clone(name));
      return bookME(name, MonitorElementData::Kind::TPROFILE2D, tprofile);
    }

    template <class ME>
    void DQMStore<ME>::enterLumi(edm::RunNumber_t run, edm::LuminosityBlockNumber_t lumi, unsigned int moduleID) {
      //TODO
    }

    template <class ME>
    MonitorElementData* DQMStore<ME>::cloneMonitorElementData(MonitorElementData const* input) {
      //TODO
      return nullptr;
    }

    template <class ME>
    void DQMStore<ME>::leaveLumi(edm::RunNumber_t run, edm::LuminosityBlockNumber_t lumi, unsigned int moduleID) {
      //TODO
    }

    template <class ME, class STORE>
    std::vector<dqm::harvesting::MonitorElement*> IGetter<ME, STORE>::getContents(std::string const& path) const {
      assert(!"NIY");
    }
    template <class ME, class STORE>
    void IGetter<ME, STORE>::getContents(std::vector<std::string>& into, bool showContents) const {
      assert(!"NIY");
    }

    template <class ME, class STORE>
    std::vector<dqm::harvesting::MonitorElement*> IGetter<ME, STORE>::getAllContents(std::string const& path) const {
      assert(!"NIY");
    }
    template <class ME, class STORE>
    std::vector<dqm::harvesting::MonitorElement*> IGetter<ME, STORE>::getAllContents(std::string const& path,
                                                                                     uint32_t runNumber,
                                                                                     uint32_t lumi) const {
      assert(!"NIY");
    }

    template <class ME, class STORE>
    ME* IGetter<ME, STORE>::get(std::string const& fullpath) const {
      assert(!"NIY");
    }

    template <class ME, class STORE>
    ME* IGetter<ME, STORE>::getElement(std::string const& path) const {
      assert(!"NIY");
    }

    template <class ME, class STORE>
    std::vector<std::string> IGetter<ME, STORE>::getSubdirs() const {
      assert(!"NIY");
    }
    template <class ME, class STORE>
    std::vector<std::string> IGetter<ME, STORE>::getMEs() const {
      assert(!"NIY");
    }
    template <class ME, class STORE>
    bool IGetter<ME, STORE>::dirExists(std::string const& path) const {
      assert(!"NIY");
    }

    template <class ME, class STORE>
    IGetter<ME, STORE>::IGetter(STORE* store) {
      store_ = store;
    }

    template <class ME>
    DQMStore<ME>::DQMStore(edm::ParameterSet const& pset, edm::ActivityRegistry&)
        : IGetter<ME, DQMStore<ME>>(this), IBooker<ME, DQMStore<ME>>(this) {}
    template <class ME>
    DQMStore<ME>::~DQMStore() {}

    template <class ME>
    void DQMStore<ME>::save(std::string const& filename,
                            std::string const& path,
                            std::string const& pattern,
                            std::string const& rewrite,
                            uint32_t run,
                            uint32_t lumi,
                            SaveReferenceTag ref,
                            int minStatus,
                            std::string const& fileupdate) {
      assert(!"NIY");
    }
    template <class ME>
    void DQMStore<ME>::savePB(std::string const& filename, std::string const& path, uint32_t run, uint32_t lumi) {
      assert(!"NIY");
    }
    template <class ME>
    bool DQMStore<ME>::open(std::string const& filename,
                            bool overwrite,
                            std::string const& path,
                            std::string const& prepend,
                            OpenRunDirs stripdirs,
                            bool fileMustExist) {
      assert(!"NIY");
    }
    template <class ME>
    bool DQMStore<ME>::load(std::string const& filename, OpenRunDirs stripdirs, bool fileMustExist) {
      assert(!"NIY");
    }

    template <class ME>
    void DQMStore<ME>::showDirStructure() const {
      assert(!"NIY");
    }

    template <class ME>
    std::vector<ME*> DQMStore<ME>::getMatchingContents(std::string const& pattern) const {
      assert(!"NIY");
    }

  }  // namespace implementation
}  // namespace dqm

template class dqm::implementation::DQMStore<dqm::legacy::MonitorElement>;

template class dqm::implementation::IGetter<dqm::legacy::MonitorElement,
                                            dqm::implementation::DQMStore<dqm::legacy::MonitorElement>>;

template class dqm::implementation::IBooker<dqm::legacy::MonitorElement,
                                            dqm::implementation::DQMStore<dqm::legacy::MonitorElement>>;
