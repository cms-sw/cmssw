// silence deprecation warnings for the DQMStore itself.
#define DQM_DEPRECATED
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/LegacyIOHelper.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/GlobalContext.h"
#include <string>
#include <regex>
#include <csignal>

#include <execinfo.h>
#include <cxxabi.h>

namespace dqm::implementation {

  std::string NavigatorBase::pwd() {
    if (cwd_.empty()) {
      return "";
    } else {
      // strip trailing slash.
      // This is inefficient and error prone (callers need to do the same
      // branching to re-add the "/"!) but some legacy code expects it like
      // that and is to complicated to change.
      assert(cwd_[cwd_.size() - 1] == '/');
      auto pwd = cwd_.substr(0, cwd_.size() - 1);
      return pwd;
    }
  }
  void NavigatorBase::cd() { setCurrentFolder(""); }
  void NavigatorBase::cd(std::string const& dir) { setCurrentFolder(dir); }
  void NavigatorBase::goUp() { cd(cwd_ + ".."); }
  void NavigatorBase::setCurrentFolder(std::string const& fullpath) {
    MonitorElementData::Path path;
    path.set(fullpath, MonitorElementData::Path::Type::DIR);
    assert(this);
    cwd_ = path.getDirname();
  }

  IBooker::IBooker(DQMStore* store) {
    store_ = store;
    scope_ = MonitorElementData::Scope::JOB;
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
                                  std::function<TH1*()> makeobject,
                                  bool forceReplace /* = false */) {
    MonitorElementData::Path path;
    std::string fullpath = cwd_ + std::string(name.View());
    path.set(fullpath, MonitorElementData::Path::Type::DIR_AND_NAME);

    // We should check if there is a local ME for this module and name already.
    // However, it is easier to do that in putME().

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

      // will be (0,0) ( = prototype) in the common case.
      // This branching is for harvesting, where we have run/lumi in the booker.
      if (this->scope_ == MonitorElementData::Scope::JOB) {
        medata.key_.id_ = edm::LuminosityBlockID();
      } else if (this->scope_ == MonitorElementData::Scope::RUN) {
        medata.key_.id_ = edm::LuminosityBlockID(this->runlumi_.run(), 0);
      } else if (this->scope_ == MonitorElementData::Scope::LUMI) {
        // In the messy case of legacy-booking a LUMI ME in beginRun (or
        // similar), where we don't have a valid lumi number yet, make sure to
        // book a prototype instead.
        if (this->runlumi_.run() != 0 && this->runlumi_.luminosityBlock() != 0) {
          medata.key_.id_ = this->runlumi_;
        } else {
          medata.key_.id_ = edm::LuminosityBlockID();
        }
      } else {
        assert(!"Illegal scope");
      }

      medata.value_.object_ = std::unique_ptr<TH1>(th1);
      MonitorElement* me_ptr = new MonitorElement(std::move(medata));
      me = store_->putME(me_ptr);
    } else {
      if (forceReplace) {
        TH1* th1 = makeobject();
        assert(th1);
        store_->debugTrackME("bookME (forceReplace)", nullptr, me);
        // surgically replace Histogram
        me->switchObject(std::unique_ptr<TH1>(th1));
      }
    }

    // me now points to a global ME owned by the DQMStore.
    assert(me);

    // each booking call returns a unique "local" ME, which the DQMStore keeps
    // in a container associated with the module (and potentially run, for
    // DQMGlobalEDAnalyzer). This will later be update to point to different
    // MEData (kept in a global ME) as needed.
    // putME creates the local ME object as needed.
    auto localme = store_->putME(me, this->moduleID_);
    // me now points to a local ME owned by the DQMStore.
    assert(localme);

    if (this->moduleID_ == 0) {
      // this is a legacy/global/harvesting booking. In this case, we return
      // the global directly. It is not advisable to hold this pointer, as we
      // may delete the global ME later, but we promise to keep it valid for
      // the entire job if there are no concurrent runs/lumis. (see
      // assertLegacySafe option).
      // We still created a local ME, so we can drive the lumi-changing for
      // legacy modules in watchPreGlobalBeginLumi.
      store_->debugTrackME("bookME (legacy)", localme, me);
      return me;
    } else {
      // the normal case.
      store_->debugTrackME("bookME (normal)", localme, me);
      return localme;
    }
  }

  MonitorElement* DQMStore::putME(MonitorElement* me) {
    auto lock = std::scoped_lock(this->booking_mutex_);
    assert(me);
    auto existing_new = globalMEs_[me->getRunLumi()].insert(me);
    if (existing_new.second == true) {
      // successfully inserted, return new object
      debugTrackME("putME (global)", nullptr, me);
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
    auto& localmes = localMEs_[moduleID];
    auto existing = localmes.find(me);
    if (existing == localmes.end()) {
      // insert new local ME
      MonitorElement* local_me = new MonitorElement(me);
      auto existing_new = localmes.insert(local_me);
      // successfully inserted, return new object
      assert(existing_new.second == true);  // insert successful
      debugTrackME("putME (local, new)", local_me, me);
      return local_me;
    } else {
      // already present, return old object
      auto local_me = *existing;
      edm::LogInfo("DQMStore") << "ME " << me->getFullname() << " booked twice in the same module.";
      // the existing local ME might not have data attached (e.g. in 2nd run)
      // in that case, we attach the global ME provided by booking above.
      // This may be a prototype or of a random run/lumi, but it ensures that
      // even LUMI histos are always valid after booking (as we promise for
      // legacy modules -- for sequential runs/lumis, there is only ever one
      // global ME, and the local one points to it).
      if (!local_me->isValid()) {
        local_me->switchData(me);
      }
      debugTrackME("putME (local, existing)", local_me, me);
      return local_me;
    }
  }

  template <typename MELIKE>
  MonitorElement* DQMStore::findME(MELIKE const& path) {
    auto lock = std::scoped_lock(this->booking_mutex_);
    for (auto& [runlumi, meset] : this->globalMEs_) {
      auto it = meset.find(path);
      if (it != meset.end()) {
        debugTrackME("findME (found)", nullptr, *it);
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

      if (!clean_trace.empty()) {
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

  void DQMStore::debugTrackME(const char* message, MonitorElement* me_local, MonitorElement* me_global) const {
    const char* scopename[] = {"INVALID", "JOB", "RUN", "LUMI"};
    if (!this->trackME_.empty() && (me_local || me_global)) {
      std::string name = me_global ? me_global->getFullname() : me_local->getFullname();
      if (name.find(this->trackME_) != std::string::npos) {
        edm::LogWarning("DQMStoreTrackME").log([&](auto& logger) {
          logger << message << " for " << name << "(" << me_local << "," << me_global << ")";
          auto writeme = [&](MonitorElement* me) {
            if (me->isValid()) {
              logger << " " << me->getRunLumi() << " scope " << scopename[me->getScope()];
              if (me->kind() >= MonitorElement::Kind::TH1F) {
                logger << " entries " << me->getEntries();
              } else if (me->kind() == MonitorElement::Kind::STRING) {
                logger << " value " << me->getStringValue();
              } else if (me->kind() == MonitorElement::Kind::REAL) {
                logger << " value " << me->getFloatValue();
              } else if (me->kind() == MonitorElement::Kind::INT) {
                logger << " value " << me->getIntValue();
              }
            } else {
              logger << " (invalid)";
            }
          };
          if (me_local) {
            logger << "  local:";
            writeme(me_local);
          }
          if (me_global) {
            logger << "  global:";
            writeme(me_global);
          }
        });
        // A breakpoint can be useful here.
        //std::raise(SIGINT);
      }
    }
  }

  MonitorElement* DQMStore::findOrRecycle(MonitorElementData::Key const& key) {
    // This is specifically for DQMRootSource, or other input modules. These
    // are special in that they use the legacy interface (no moduleID, no local
    // MEs) but need to be able to handle concurrent lumisections correctly.
    // The logic is very similar to that in enterLumi; this is enterLumi for
    // Input Modules.
    auto lock = std::scoped_lock(this->booking_mutex_);
    auto existing = this->get(key);
    if (existing) {
      // exactly matching ME found, needs merging with the new data.
      debugTrackME("findOrRecycle (found)", nullptr, existing);
      return existing;
    }  // else

    // this is where we'd expect the ME.
    auto& targetset = this->globalMEs_[key.id_];
    // this is where we can get MEs to reuse.
    auto& prototypes = this->globalMEs_[edm::LuminosityBlockID()];

    auto proto = prototypes.find(key.path_);
    if (proto != prototypes.end()) {
      MonitorElement* oldme = *proto;
      assert(oldme->getScope() == key.scope_);
      prototypes.erase(proto);
      auto medata = oldme->release(/* expectOwned */ true);  // destroy the ME, get its data.
      // in this situation, nobody should be filling the ME concurrently.
      medata->data_.key_.id_ = key.id_;
      // We reuse the ME object here, even if we don't have to. This ensures
      // that when running single-threaded without concurrent lumis/runs,
      // the global MEs will also live forever and allow legacy usages.
      oldme->switchData(medata);
      auto result = targetset.insert(oldme);
      assert(result.second);       // was new insertion
      auto newme = *result.first;  // iterator to new ME
      assert(oldme == newme);      // recycling!
      // newme is reset and ready to accept data.
      debugTrackME("findOrRecycle (recycled)", nullptr, newme);
      return newme;
    }  // else

    return nullptr;
  }

  void DQMStore::initLumi(edm::RunNumber_t run, edm::LuminosityBlockNumber_t lumi) {
    // Call initLumi for all modules, as a global operation.
    auto lock = std::scoped_lock(this->booking_mutex_);
    for (auto& kv : this->localMEs_) {
      initLumi(run, lumi, kv.first);
    }
  }

  void DQMStore::initLumi(edm::RunNumber_t run, edm::LuminosityBlockNumber_t lumi, uint64_t moduleID) {
    // Make sure global MEs for the run/lumi exist (depending on scope)

    auto lock = std::scoped_lock(this->booking_mutex_);

    // these are the MEs we need to update.
    auto& localset = this->localMEs_[moduleID];
    // this is where they need to point to.
    // This could be a per-run or per-lumi set (depending on lumi == 0)
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
        debugTrackME("initLumi (existing)", nullptr, *target);
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
          debugTrackME("initLumi (reused)", nullptr, *target);
        } else {
          // no prototype available. That means we have concurrent Lumis/Runs,
          // and need to make a clone now.
          auto anyme = this->findME(me);
          assert(anyme || !"local ME without any global ME!");
          if (checkScope(anyme->getScope()) == false) {
            continue;
          }  // else

          // whenever we clone global MEs, it is no longer safe to hold
          // pointers to them.
          assert(!assertLegacySafe_);

          MonitorElementData newdata = anyme->cloneMEData();
          newdata.key_.id_ = edm::LuminosityBlockID(run, lumi);
          auto newme = new MonitorElement(std::move(newdata));
          newme->Reset();  // we cloned a ME in use, not an empty prototype
          auto result = targetset.insert(newme);
          assert(result.second);  // was new insertion
          target = result.first;  // iterator to new ME
          debugTrackME("initLumi (allocated)", nullptr, *target);
        }
      }
    }
  }

  void DQMStore::enterLumi(edm::RunNumber_t run, edm::LuminosityBlockNumber_t lumi, uint64_t moduleID) {
    // point the local MEs for this module to these global MEs.

    // This needs to happen before we can use the global MEs for this run/lumi here.
    // We could do it lazyly here, or eagerly globally in global begin lumi.
    //initLumi(run, lumi, moduleID);

    auto lock = std::scoped_lock(this->booking_mutex_);

    // these are the MEs we need to update.
    auto& localset = this->localMEs_[moduleID];
    // this is where they need to point to.
    auto& targetset = this->globalMEs_[edm::LuminosityBlockID(run, lumi)];

    // only for a sanity check
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
      if (target == targetset.end()) {
        auto anyme = this->findME(me);
        debugTrackME("enterLumi (nothingtodo)", me, nullptr);
        assert(anyme && checkScope(anyme->getScope()) == false);
        continue;
      }
      assert(target != targetset.end());  // initLumi should have taken care of this.
      // now we have the proper global ME in the right place, point the local there.
      // This is only safe if the name is exactly the same -- else it might corrupt
      // the tree structure of the set!
      me->switchData(*target);
      debugTrackME("enterLumi (switchdata)", me, *target);
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
      // we have to be very careful with the ME here, it might not be backed by data at all.
      if (me->isValid() && checkScope(me->getScope()) == true) {
        // if we left the scope, simply release the data.
        debugTrackME("leaveLumi (release)", me, nullptr);
        me->release(/* expectOwned */ false);
      }
    }
  }

  void DQMStore::cleanupLumi(edm::RunNumber_t run, edm::LuminosityBlockNumber_t lumi) {
    // now, we are done with the lumi, no modules have any work to do on these
    // MEs, and the output modules have saved this lumi/run. Remove/recycle
    // the MEs here.

    auto lock = std::scoped_lock(this->booking_mutex_);

    // in case of end-job cleanup we need different logic because of the
    // prototype set.
    assert(run != 0 || lumi != 0);
    auto& prototypes = this->globalMEs_[edm::LuminosityBlockID()];

    // these are the MEs we need to get rid of...
    auto meset = std::set<MonitorElement*, MonitorElement::MEComparison>();
    // ... we take them out first.
    meset.swap(this->globalMEs_[edm::LuminosityBlockID(run, lumi)]);

    // temporary buffer for the MEs to recycle, we must not change the key
    // while they are in a set.
    auto torecycle = std::vector<MonitorElement*>();

    // here, this is only a sanity check and not functionally needed.
    auto checkScope = [run, lumi](MonitorElementData::Scope scope) {
      if (scope == MonitorElementData::Scope::JOB) {
        assert(run == 0 && lumi == 0);
      } else if (scope == MonitorElementData::Scope::RUN) {
        assert(run != 0 && lumi == 0);
      } else if (scope == MonitorElementData::Scope::LUMI) {
        assert(lumi != 0);
      } else {
        assert(!"Impossible Scope.");
      }
    };

    for (MonitorElement* me : meset) {
      assert(me->isValid());       // global MEs should always be valid.
      checkScope(me->getScope());  // we should only see MEs of one scope here.
      auto other = this->findME(me);
      if (other) {
        // we still have a global one, so we can just remove this.
        debugTrackME("cleanupLumi (delete)", nullptr, me);
        delete me;
      } else {
        // we will modify the ME, so it needs to be out of the set.
        // use a temporary vector to be save.
        debugTrackME("cleanupLumi (recycle)", nullptr, me);
        torecycle.push_back(me);
      }
    }

    meset.clear();

    for (MonitorElement* me : torecycle) {
      auto medata = me->release(/* expectOwned */ true);  // destroy the ME, get its data.
      medata->data_.key_.id_ = edm::LuminosityBlockID();  // prototype
      // We reuse the ME object here, even if we don't have to. This ensures
      // that when running single-threaded without concurrent lumis/runs,
      // the global MEs will also live forever and allow legacy usages.
      me->switchData(medata);
      // reset here (not later) to still catch random legacy fill calls.
      me->Reset();
      auto result = prototypes.insert(me);
      assert(result.second);  // was new insertion, else findME should succeed
      debugTrackME("cleanupLumi (reset)", nullptr, me);
    }
  }

  std::vector<dqm::harvesting::MonitorElement*> IGetter::getContents(std::string const& pathname) const {
    auto lock = std::scoped_lock(store_->booking_mutex_);
    std::vector<MonitorElement*> out;
    MonitorElementData::Path path;
    path.set(pathname, MonitorElementData::Path::Type::DIR);
    for (auto& [runlumi, meset] : store_->globalMEs_) {
      auto it = meset.lower_bound(path);
      while (it != meset.end() && (*it)->getPathname() == path.getDirname()) {
        store_->debugTrackME("getContents (match)", nullptr, *it);
        out.push_back(*it);
        ++it;
      }
    }
    return out;
  }

  std::vector<dqm::harvesting::MonitorElement*> IGetter::getAllContents(std::string const& pathname) const {
    auto lock = std::scoped_lock(store_->booking_mutex_);
    std::vector<MonitorElement*> out;
    MonitorElementData::Path path;
    path.set(pathname, MonitorElementData::Path::Type::DIR);
    // make sure this is normalized by getting it from Path object.
    auto path_str = path.getFullname();
    for (auto& [runlumi, meset] : store_->globalMEs_) {
      auto it = meset.lower_bound(path);
      // rfind can be used as a prefix match.
      while (it != meset.end() && (*it)->getPathname().rfind(path_str, 0) == 0) {
        if (runlumi == edm::LuminosityBlockID() && (*it)->getScope() != MonitorElementData::Scope::JOB) {
          // skip prototypes
        } else {
          store_->debugTrackME("getAllContents (match)", nullptr, *it);
          out.push_back(*it);
        }
        ++it;
      }
    }
    return out;
  }
  std::vector<dqm::harvesting::MonitorElement*> IGetter::getAllContents(std::string const& pathname,
                                                                        uint32_t runNumber,
                                                                        uint32_t lumi) const {
    auto lock = std::scoped_lock(store_->booking_mutex_);
    std::vector<MonitorElement*> out;
    MonitorElementData::Path path;
    path.set(pathname, MonitorElementData::Path::Type::DIR);
    // make sure this is normalized by getting it from Path object.
    auto path_str = path.getFullname();
    auto const& meset = store_->globalMEs_[edm::LuminosityBlockID(runNumber, lumi)];
    auto it = meset.lower_bound(path);

    // decide if the ME should be save din DQMIO based on the list provided
    bool saveIt = true;

    // rfind can be used as a prefix match.
    while (it != meset.end() && (*it)->getFullname().rfind(path_str, 0) == 0) {
      if (store_->doSaveByLumi_ && not store_->MEsToSave_.empty()) {
        for (std::vector<std::string>::const_iterator ipath = store_->MEsToSave_.begin();
             ipath != store_->MEsToSave_.end();
             ++ipath) {
          std::string name = (*it)->getFullname();
          if (name.find(*ipath) != std::string::npos) {
            saveIt = true;
            //std::cout<<name<<" compared to"<<ipath->data()<<std::endl;
            break;
          }
          saveIt = false;
        }
      }

      store_->debugTrackME("getAllContents (run/lumi match)", nullptr, *it);
      if (saveIt) {
        out.push_back(*it);
        if (store_->doSaveByLumi_)
          store_->debugTrackME("getAllContents (run/lumi saved)", nullptr, *it);
      }
      ++it;
    }
    return out;
  }

  MonitorElement* IGetter::get(std::string const& fullpath) const {
    MonitorElementData::Path path;
    path.set(fullpath, MonitorElementData::Path::Type::DIR_AND_NAME);
    // this only really makes sense if there is only one instance of this ME,
    // but the signature of this method also only makes sense in that case.
    return store_->findME(path);
  }

  MonitorElement* IGetter::get(MonitorElementData::Key const& key) const {
    auto const& meset = store_->globalMEs_[key.id_];
    auto it = meset.find(key.path_);
    if (it != meset.end()) {
      assert((*it)->getScope() == key.scope_);
      store_->debugTrackME("get (key found)", nullptr, *it);
      return *it;
    }
    return nullptr;
  }

  MonitorElement* IGetter::getElement(std::string const& path) const {
    auto result = this->get(path);
    if (result == nullptr) {
      throw cms::Exception("iGetter Error") << "ME " << path << " was requested but not found.";
    }
    return result;
  }

  std::vector<std::string> IGetter::getSubdirs() const {
    // This is terribly inefficient, esp. if this method is then used to
    // recursively enumerate whatever getAllContents would return anyways.
    // But that is fine, any such code should just use getAllContents instead.
    std::set<std::string> subdirs;
    for (auto me : this->getAllContents(this->cwd_)) {
      const auto& name = me->getPathname();
      auto subdirname = name.substr(this->cwd_.length(), std::string::npos);
      auto dirname = subdirname.substr(0, subdirname.find('/'));
      subdirs.insert(dirname);
    }
    std::vector<std::string> out;
    for (const auto& dir : subdirs) {
      if (dir.length() == 0)
        continue;
      out.push_back(this->cwd_ + dir);
    }
    return out;
  }

  std::vector<std::string> IGetter::getMEs() const {
    auto mes = this->getContents(this->cwd_);
    std::vector<std::string> out;
    out.reserve(mes.size());
    for (auto me : mes) {
      out.push_back(me->getName());
    }
    return out;
  }

  bool IGetter::dirExists(std::string const& path) const {
    // we don't claim this is fast.
    return !this->getAllContents(path).empty();
  }

  IGetter::IGetter(DQMStore* store) { store_ = store; }

  IGetter::~IGetter() {}

  DQMStore::DQMStore(edm::ParameterSet const& pset, edm::ActivityRegistry& ar) : IGetter(this), IBooker(this) {
    verbose_ = pset.getUntrackedParameter<int>("verbose", 0);
    assertLegacySafe_ = pset.getUntrackedParameter<bool>("assertLegacySafe", true);
    doSaveByLumi_ = pset.getUntrackedParameter<bool>("saveByLumi", false);
    MEsToSave_ = pset.getUntrackedParameter<std::vector<std::string>>("MEsToSave", std::vector<std::string>());
    trackME_ = pset.getUntrackedParameter<std::string>("trackME", "");

    // Set lumi and run for legacy booking.
    // This is no more than a guess with concurrent runs/lumis, but should be
    // correct for purely sequential legacy stuff.
    // Also reset Scope, such that legacy modules can expect it to be JOB.
    // initLumi and leaveLumi are needed for all module types: these handle
    // creating and deleting global MEs as needed, which has to happen even if
    // a module does not see lumi transitions.
    ar.watchPreGlobalBeginRun([this](edm::GlobalContext const& gc) {
      this->setRunLumi(gc.luminosityBlockID());
      this->initLumi(gc.luminosityBlockID().run(), /* lumi */ 0);
      this->enterLumi(gc.luminosityBlockID().run(), /* lumi */ 0, /* moduleID */ 0);
      this->setScope(MonitorElementData::Scope::JOB);
    });
    ar.watchPreGlobalBeginLumi([this](edm::GlobalContext const& gc) {
      this->setRunLumi(gc.luminosityBlockID());
      this->initLumi(gc.luminosityBlockID().run(), gc.luminosityBlockID().luminosityBlock());
      this->enterLumi(gc.luminosityBlockID().run(), gc.luminosityBlockID().luminosityBlock(), /* moduleID */ 0);
    });
    ar.watchPostGlobalEndRun([this](edm::GlobalContext const& gc) {
      this->leaveLumi(gc.luminosityBlockID().run(), /* lumi */ 0, /* moduleID */ 0);
    });
    ar.watchPostGlobalEndLumi([this](edm::GlobalContext const& gc) {
      this->leaveLumi(gc.luminosityBlockID().run(), gc.luminosityBlockID().luminosityBlock(), /* moduleID */ 0);
    });

    // Trigger cleanup after writing. This is needed for all modules; we can
    // only run the cleanup after all output modules have run.
    ar.watchPostGlobalWriteLumi([this](edm::GlobalContext const& gc) {
      this->cleanupLumi(gc.luminosityBlockID().run(), gc.luminosityBlockID().luminosityBlock());
    });
    ar.watchPostGlobalWriteRun(
        [this](edm::GlobalContext const& gc) { this->cleanupLumi(gc.luminosityBlockID().run(), 0); });

    // no cleanup at end of job, we don't really need it.
  }

  DQMStore::~DQMStore() {}

  void DQMStore::save(std::string const& filename, std::string const& path) {
    LegacyIOHelper h(this);
    // no run number passed, will save a flat ROOT file (rather than 'Run xxxxxx/.../Run Summary/...')
    h.save(filename, path);
  }

  bool DQMStore::open(std::string const& filename,
                      bool overwrite,
                      std::string const& path,
                      std::string const& prepend,
                      OpenRunDirs stripdirs,
                      bool fileMustExist) {
    assert(!"NIY");
  }

}  // namespace dqm::implementation
