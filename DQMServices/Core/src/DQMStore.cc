// silence deprecation warnings for the DQMStore itself.
#define DQM_DEPRECATED
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <string>
#include <regex>
#include <csignal>

#include <execinfo.h>
#include <cxxabi.h>

namespace dqm::implementation {

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

  IBooker::IBooker(DQMStore* store) {
    store_ = store;
    scope_ = MonitorElementData::Scope::RUN;
  }

  IBooker::~IBooker() {}

  MonitorElementData::Scope IBooker::setScope(MonitorElementData::Scope newscope) {
    auto oldscope = scope_;
    scope_ = newscope;
    return oldscope;
  }
  uint64_t MonitorElementData::setModuleID(uint64_t moduleID) {
    auto oldid = moduleID_;
    moduleID_ = moduleID;
    return oldid;
  }

  edm::LuminosityBlockID setRunLumi(edm::LuminosityBlockID runlumi) {
    auto oldrunlumi = runlumi_;
    runlumi_ = runlumi;
    return oldrunlumi;
  }

  MonitorElement* IBooker::bookME(TString const& name,
                                  MonitorElementData::Kind kind,
                                  std::function<TH1*()> makeobject) {
    MonitorElementData::Path path;
    std::string fullpath = pwd() + std::string(name.View());
    path.set(fullpath, MonitorElementData::Path::Type::DIR_AND_NAME);
    MonitorElement* me store_->findME(path);
    printTrace("Booking " + std::string(name) + (me ? " (existing)" : " (new)")); 
    if (me == nullptr) {
      // no existing global ME found. We need to instantiate one, and put it
      // into the DQMStore. This will typically be a prototype, unless run and
      // lumi are set and we proces a legacy booking call.
      TH1* th1 = makeobject();
      MonitorElementData medata;
      medata.key_.path_ = path;
      medata.key_.kind_ = kind;
      medata.key_.scope_ = this->scope_;
      // will be 0 ( = prototype) in the common case.
      medata.key_.id_ = edm::LuminosityBlockID(this->run_, this_->lumi_);
      medata.value_.object_ = std::unique_ptr<TH1>(th1);
      MonitorElement* me_ptr = new MonitorElement(data, /* is_owned */ true, /* is_readonly */ false);
      me = store_->putME(me_ptr);
    }
    // me now points to a global ME owned by the DQMStore.
    assert(me);

    // each booking call returns a unique "local" ME, which the DQMStore keeps
    // in a container associated with the module (and potentially run, for
    // DQMGlobalEDAnalyzer). This will later be update to point to different
    // MEData (kept in a global ME) as needed.
    MonitorElement* local_me = new MonitorElement(me);
    me = store_->putME(local_me);
    // me now points to a local ME owned by the DQMStore.
    assert(me);
    return me;
  }

  MonitorElement* DQMStore::putME(MonitorElement* me) {
    assert(me);
    auto existing_new = globalMEs_[me->getRunLumi()].insert(me);
    if (existing_new.second = true) {
      // successfully inserted, return new object
      return me;
    } else {
      // already present, return old object
      delete me;
      assert(!"Currently, this should never happen.");
      return *(existing_new.first);
    }
  }

  MonitorElement* DQMStore::putME(MonitorElement* me, uint64_t moduleID) {
    assert(me);
    auto existing_new = localMEs_[moduleID].insert(me);
    if (existing_new.second = true) {
      // successfully inserted, return new object
      return me;
    } else {
      // already present, return old object
      delete me;
      assert(!"Currently, this should never happen.");
      return *(existing_new.first);
    }
  }

  void DQMStore::printTrace(std::string const& message) {
    if (verbose_ < 3) return;
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

  void DQMStore::enterLumi(edm::RunNumber_t run, edm::LuminosityBlockNumber_t lumi, uint64_t moduleID) {
    //TODO
  }

  MonitorElementData* DQMStore::cloneMonitorElementData(MonitorElementData const* input) {
    //TODO
    return nullptr;
  }

  void DQMStore::leaveLumi(edm::RunNumber_t run, edm::LuminosityBlockNumber_t lumi, uint64_t moduleID) {
    //TODO
  }

  std::vector<dqm::harvesting::MonitorElement*> IGetter::getContents(std::string const& path) const { assert(!"NIY"); }
  void IGetter::getContents(std::vector<std::string>& into, bool showContents) const { assert(!"NIY"); }

  std::vector<dqm::harvesting::MonitorElement*> IGetter::getAllContents(std::string const& path) const {
    assert(!"NIY");
  }
  std::vector<dqm::harvesting::MonitorElement*> IGetter::getAllContents(std::string const& path,
                                                                        uint32_t runNumber,
                                                                        uint32_t lumi) const {
    assert(!"NIY");
  }

  MonitorElement* IGetter::get(std::string const& fullpath) const { assert(!"NIY"); }

  MonitorElement* IGetter::getElement(std::string const& path) const { assert(!"NIY"); }

  std::vector<std::string> IGetter::getSubdirs() const { assert(!"NIY"); }
  std::vector<std::string> IGetter::getMEs() const { assert(!"NIY"); }
  bool IGetter::dirExists(std::string const& path) const { assert(!"NIY"); }

  IGetter::IGetter(DQMStore* store) { store_ = store; }

  IGetter::~IGetter() {}

  DQMStore::DQMStore(edm::ParameterSet const& pset, edm::ActivityRegistry&) : IGetter(this), IBooker(this) {
    verbose_ = pset.getUntrackedParameter<int>("verbose", 0);
  }

  DQMStore::~DQMStore() {}

  void DQMStore::save(std::string const& filename,
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
  void DQMStore::savePB(std::string const& filename, std::string const& path, uint32_t run, uint32_t lumi) {
    assert(!"NIY");
  }
  bool DQMStore::open(std::string const& filename,
                      bool overwrite,
                      std::string const& path,
                      std::string const& prepend,
                      OpenRunDirs stripdirs,
                      bool fileMustExist) {
    assert(!"NIY");
  }
  bool DQMStore::load(std::string const& filename, OpenRunDirs stripdirs, bool fileMustExist) { assert(!"NIY"); }

  void DQMStore::showDirStructure() const { assert(!"NIY"); }

  std::vector<MonitorElement*> DQMStore::getMatchingContents(std::string const& pattern) const { assert(!"NIY"); }

}  // namespace dqm::implementation
