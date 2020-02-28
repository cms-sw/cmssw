
#include "DataFormats/Common/interface/HLTenums.h"
#include "DataFormats/HLTReco/interface/HLTPerformanceInfo.h"
#include <algorithm>

HLTPerformanceInfo::HLTPerformanceInfo() {
  paths_.clear();
  modules_.clear();
}

HLTPerformanceInfo::PathList::iterator HLTPerformanceInfo::findPath(const char* pathName) {
  PathList::iterator l = std::find(paths_.begin(), paths_.end(), pathName);
  return l;
}

HLTPerformanceInfo::Modules::iterator HLTPerformanceInfo::findModule(const char* moduleInstanceName) {
  return std::find(modules_.begin(), modules_.end(), moduleInstanceName);
}

double HLTPerformanceInfo::totalTime() const {
  double t = 0;
  for (size_t i = 0; i < modules_.size(); ++i) {
    t += modules_[i].time();
  }
  return t;
}

double HLTPerformanceInfo::totalCPUTime() const {
  double t = 0;
  for (size_t i = 0; i < modules_.size(); ++i) {
    t += modules_[i].cputime();
  }
  return t;
}

double HLTPerformanceInfo::totalPathTime(const size_t pathnumber) {
  double t = 0;
  unsigned int cnt = 0;
  ModulesInPath::const_iterator i = paths_[pathnumber].begin();
  for (; i != paths_[pathnumber].end(); ++i) {
    if (cnt > paths_[pathnumber].status().index())
      break;
    ++cnt;
    t += modules_[*i].time();
  }
  return t;
}

double HLTPerformanceInfo::totalPathCPUTime(const size_t pathnumber) {
  double t = 0;
  unsigned int cnt = 0;
  ModulesInPath::const_iterator i = paths_[pathnumber].begin();
  for (; i != paths_[pathnumber].end(); ++i) {
    if (cnt > paths_[pathnumber].status().index())
      break;
    ++cnt;
    t += modules_[*i].cputime();
  }
  return t;
}

double HLTPerformanceInfo::longestModuleTime() const {
  double t = -1;
  for (Modules::const_iterator i = modules_.begin(); i != modules_.end(); ++i) {
    t = std::max(i->time(), t);
  }
  return t;
}

double HLTPerformanceInfo::longestModuleCPUTime() const {
  double t = -1;
  for (Modules::const_iterator i = modules_.begin(); i != modules_.end(); ++i) {
    t = std::max(i->cputime(), t);
  }
  return t;
}

std::string HLTPerformanceInfo::longestModuleTimeName() const {
  double t = -1;
  std::string slowpoke("unknown");
  for (Modules::const_iterator i = modules_.begin(); i != modules_.end(); ++i) {
    if (i->time() > t) {
      slowpoke = i->name();
      t = i->time();
    }
  }
  return slowpoke;
}

std::string HLTPerformanceInfo::longestModuleCPUTimeName() const {
  double t = -1;
  std::string slowpoke("unknown");
  for (Modules::const_iterator i = modules_.begin(); i != modules_.end(); ++i) {
    if (i->cputime() > t) {
      slowpoke = i->name();
      t = i->cputime();
    }
  }
  return slowpoke;
}

// I think we can no longer do this as it requires going from path -> modules
// int HLTPerformanceInfo::Module::indexInPath(Path path) const
// {
//   int ctr = 0 ;
//   ModulesInPath::const_iterator iter = path.begin();
//   for ( ; iter != path.end(); ++iter ) {
//     if (modules_[*iter].name() == this->name()) return ctr ;
//     ctr++ ;
//   }
//   //--- Module not found in path ---
//   return -1 ;
// }

const HLTPerformanceInfo::Module& HLTPerformanceInfo::getModuleOnPath(size_t m, size_t p) const {
  // well if this doesn't get your attention....
  assert(p < paths_.size() && m < paths_[p].numberOfModules());
  size_t j = paths_[p].getModuleIndex(m);
  return modules_.at(j);
}

bool HLTPerformanceInfo::uniqueModule(const char* mod) const {
  int mCtr = 0;
  for (size_t p = 0; p < paths_.size(); ++p) {
    for (size_t m = 0; m < paths_[p].numberOfModules(); ++m) {
      size_t modIndex = paths_[p].getModuleIndex(m);
      if (modules_[modIndex].name() == std::string(mod))
        ++mCtr;
      if (mCtr > 1)
        return false;
    }
  }

  if (mCtr == 0)
    return false;
  return true;
}

int HLTPerformanceInfo::moduleIndexInPath(const char* mod, const char* path) {
  PathList::iterator p = findPath(path);
  if (p == endPaths())
    return -1;  // Path doesn't exist
  int ctr = 0;
  for (ModulesInPath::const_iterator j = p->begin(); j != p->end(); ++j) {
    if (modules_.at(*j) == mod)
      return ctr;
    ctr++;
  }
  return -2;  // module not found on path
}

// Set the status of the module based on the path's status
// make sure not to wipe out ones that come after the last
// module run on the particular path
void HLTPerformanceInfo::setStatusOfModulesFromPath(const char* pathName) {
  PathList::iterator p = findPath(pathName);
  if (p == endPaths()) {
    return;  // do nothing
  }
  unsigned int ctr = 0;
  for (ModulesInPath::const_iterator j = p->begin(); j != p->end(); ++j) {
    edm::hlt::HLTState modState = edm::hlt::Ready;
    unsigned int modIndex = 0;

    // get module in the master list
    Module& mod = modules_.at(*j);

    if (!mod.status().wasrun()) {
      if (p->status().accept()) {
        modState = edm::hlt::Pass;
      } else {
        if (p->status().index() > ctr) {
          modState = edm::hlt::Pass;
        } else if (p->status().index() == ctr) {
          modState = p->status().state();
        }
      }
      mod.setStatus(edm::HLTPathStatus(modState, modIndex));
    }
    ctr++;
  }
}
