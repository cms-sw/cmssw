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
  uint64_t IBooker::setModuleID(uint64_t moduleID) {
    auto oldid = moduleID_;
    moduleID_ = moduleID;
    return oldid;
  }

  edm::LuminosityBlockID IBooker::setRunLumi(edm::LuminosityBlockID runlumi) {
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
    MonitorElement* me = store_->findME(path);
    store_->printTrace("Booking " + std::string(name) + (me ? " (existing)" : " (new)"));
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
      medata.key_.id_ = this->runlumi_;
      medata.value_.object_ = std::unique_ptr<TH1>(th1);
      MonitorElement* me_ptr = new MonitorElement(std::move(medata));
      me = store_->putME(me_ptr);
    }
    // me now points to a global ME owned by the DQMStore.
    assert(me);

    // each booking call returns a unique "local" ME, which the DQMStore keeps
    // in a container associated with the module (and potentially run, for
    // DQMGlobalEDAnalyzer). This will later be update to point to different
    // MEData (kept in a global ME) as needed.
    MonitorElement* local_me = new MonitorElement(me);
    me = store_->putME(local_me, this->moduleID_);
    // me now points to a local ME owned by the DQMStore.
    assert(me);
    // TODO: maybe return global ME for legacy/harvesting bookings.
    return me;
  }

  MonitorElement* DQMStore::putME(MonitorElement* me) {
    auto lock = std::scoped_lock(this->booking_mutex_);
    assert(me);
    auto existing_new = globalMEs_[me->getRunLumi()].insert(me);
    if (existing_new.second == true) {
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
    auto lock = std::scoped_lock(this->booking_mutex_);
    assert(me);
    auto existing_new = localMEs_[moduleID].insert(me);
    if (existing_new.second == true) {
      // successfully inserted, return new object
      return me;
    } else {
      // already present, return old object
      delete me;
      assert(!"Currently, this should never happen.");
      return *(existing_new.first);
    }
  }

  template <typename MELIKE>
  MonitorElement* DQMStore::findME(MELIKE const& path) {
    auto lock = std::scoped_lock(this->booking_mutex_);
    for (auto& [runlumi, meset] : this->globalMEs_) {
      auto it = meset.find(path);
      if (it != meset.end()) {
        // no guarantee on which ME we return here -- only that clone'ing this
        // would give a valid ME for that path.
        return *it;
      }
    }
    return nullptr;
  }

  void DQMStore::printTrace(std::string const& message) {
    if (verbose_ < 3)
      return;
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
    // Make sure global MEs for the run/lumi exist (depending on scope), and
    // point the local MEs for this module to these global MEs.

    auto lock = std::scoped_lock(this->booking_mutex_);

    // these are the MEs we need to update.
    auto& localset = this->localMEs_[moduleID];
    // this is where they need to point to.
    auto& targetset = this->globalMEs_[edm::LuminosityBlockID(run, lumi)];
    // this is where we can get MEs to reuse.
    auto& prototypes = this->globalMEs_[edm::LuminosityBlockID()];

    auto checkScope = [run, lumi](MonitorElementData::Scope scope) {
      if (scope == MonitorElementData::Scope::JOB) {
        return (run == 0 && lumi == 0);
      } else if (scope == MonitorElementData::Scope::RUN) {
        return (run != 0 && lumi == 0);
      } else if (scope == MonitorElementData::Scope::LUMI) {
        return (lumi != 0);
      }
      assert(!"Impossible Scope.");
      return false;
    };

    for (MonitorElement* me : localset) {
      auto target = targetset.find(me);  // lookup by path, thanks to MEComparison
      if (target != targetset.end()) {
        // we already have a ME, just use it!
      } else {
        // look for a prototype to reuse.
        auto proto = prototypes.find(me);
        if (proto != prototypes.end()) {
          // first, check if this ME needs updating at all. We can only check
          // the scope once we have an actual global ME instance, the local ME
          // might not have any data attached!
          if (checkScope((*proto)->getScope()) == false) {
            continue;
          }  // else
          // reuse that.
          MonitorElement* oldme = *proto;
          prototypes.erase(proto);
          auto medata = oldme->release(/* expectOwned */ true);  // destroy the ME, get its data.
          // in this situation, nobody should be filling the ME concurrently.
          medata->data_.key_.id_ = edm::LuminosityBlockID(run, lumi);
          // We reuse the ME object here, even if we don't have to. This ensures
          // that when running single-threaded without concurrent lumis/runs,
          // the global MEs will also live forever and allow legacy usages.
          oldme->switchData(medata);
          auto result = targetset.insert(oldme);
          assert(result.second);  // was new insertion
          target = result.first;  // iterator to new ME
        } else {
          // no prototype available. That means we have concurrent Lumis/Runs,
          // and need to make a clone now.
          auto anyme = this->findME(me);
          assert(anyme || !"local ME without any global ME!");
          if (checkScope(anyme->getScope()) == false) {
            continue;
          }  // else
          MonitorElementData newdata = anyme->cloneMEData();
          auto newme = new MonitorElement(std::move(newdata));
          newme->Reset();  // we cloned a ME in use, not an empty prototype
          auto result = targetset.insert(newme);
          assert(result.second);  // was new insertion
          target = result.first;  // iterator to new ME
        }
      }
      // now we have the proper global ME in the right place, point the local there.
      // This is only safe if the name is exactly the same -- else it might corrupt
      // the tree structure of the set!
      me->switchData(*target);
    }
  }

  void DQMStore::leaveLumi(edm::RunNumber_t run, edm::LuminosityBlockNumber_t lumi, uint64_t moduleID) {
    // here, we remove the pointers in the local MEs. No deletion or recycling
    // yet -- this has to happen after the output module had a chance to do its
    // work. We just leave the global MEs where they are. This is purely an
    // accounting step, the cleanup code has to check that nobody is using the
    // ME any more, and here we make sure that is the case.

    auto lock = std::scoped_lock(this->booking_mutex_);

    // these are the MEs we need to update.
    auto& localset = this->localMEs_[moduleID];

    auto checkScope = [run, lumi](MonitorElementData::Scope scope) {
      if (scope == MonitorElementData::Scope::JOB) {
        return (run == 0 && lumi == 0);
      } else if (scope == MonitorElementData::Scope::RUN) {
        return (run != 0 && lumi == 0);
      } else if (scope == MonitorElementData::Scope::LUMI) {
        return (lumi != 0);
      }
      assert(!"Impossible Scope.");
      return false;
    };

    for (MonitorElement* me : localset) {
      if (checkScope(me->getScope()) == true) {
        // if we left the scope, simply release the data.
        me->release(/* expectOwned */ false);
      }
    }
  }

  std::vector<dqm::harvesting::MonitorElement*> IGetter::getContents(std::string const& pathname) const {
    std::vector<MonitorElement*> out;
    MonitorElementData::Path path;
    path.set(pathname, MonitorElementData::Path::Type::DIR);
    for (auto& [runlumi, meset] : store_->globalMEs_) {
      auto it = meset.lower_bound(path);
      // rfind can be used as a prefix match.
      while (it != meset.end() && (*it)->getPathname() == path.getDirname()) {
        out.push_back(*it);
        ++it;
      }
    }
    return out;
  }

  void IGetter::getContents(std::vector<std::string>& into, bool showContents) const { assert(!"NIY"); }

  std::vector<dqm::harvesting::MonitorElement*> IGetter::getAllContents(std::string const& pathname) const {
    std::vector<MonitorElement*> out;
    MonitorElementData::Path path;
    path.set(pathname, MonitorElementData::Path::Type::DIR);
    // make sure this is normalized by getting it from Path object.
    auto path_str = path.getFullname();
    for (auto& [runlumi, meset] : store_->globalMEs_) {
      auto it = meset.lower_bound(path);
      // rfind can be used as a prefix match.
      while (it != meset.end() && (*it)->getPathname().rfind(path_str, 0) == 0) {
        out.push_back(*it);
        ++it;
      }
    }
    return out;
  }
  std::vector<dqm::harvesting::MonitorElement*> IGetter::getAllContents(std::string const& pathname,
                                                                        uint32_t runNumber,
                                                                        uint32_t lumi) const {
    std::vector<MonitorElement*> out;
    MonitorElementData::Path path;
    path.set(pathname, MonitorElementData::Path::Type::DIR);
    // make sure this is normalized by getting it from Path object.
    auto path_str = path.getFullname();
    auto meset = store_->globalMEs_[edm::LuminosityBlockID(runNumber, lumi)];
    auto it = meset.lower_bound(path);
    // rfind can be used as a prefix match.
    while (it != meset.end() && (*it)->getFullname().rfind(path_str, 0) == 0) {
      out.push_back(*it);
      ++it;
    }
    return out;
  }

  MonitorElement* IGetter::get(std::string const& fullpath) const {
    MonitorElementData::Path path;
    path.set(fullpath, MonitorElementData::Path::Type::DIR_AND_NAME);
    // this only really makes sense if there is only one instance of this ME,
    // but the signature of this mthod also only makes sense in that case.
    return store_->findME(path);
  }

  MonitorElement* IGetter::getElement(std::string const& path) const {
    auto result = this->get(path);
    if (result == nullptr) {
      throw cms::Exception("iGetter Error") << "ME " << path << " was requested but not found.";
    }
    return result;
  }

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
