// $Id: HLTPerformanceInfo.cc,v 1.12 2008/07/24 16:22:58 wittich Exp $

#include "DataFormats/Common/interface/HLTenums.h"
#include "DataFormats/HLTReco/interface/HLTPerformanceInfo.h"

HLTPerformanceInfo::HLTPerformanceInfo() {
  paths_.clear(); modules_.clear();
}

HLTPerformanceInfo::PathList::iterator 
HLTPerformanceInfo::findPath(const char* pathName) {
  PathList::iterator l = std::find(paths_.begin(), paths_.end(),
					 pathName);
  return l; 
}

HLTPerformanceInfo::Modules::iterator 
HLTPerformanceInfo::findModule(const char* moduleInstanceName) {
  return std::find(modules_.begin(), modules_.end(),
		   moduleInstanceName);
}



double HLTPerformanceInfo::totalTime() const {
  double t = 0;
  for ( size_t i = 0; i < modules_.size(); ++i ) {
    t += modules_[i].time();
  }
  return t;
}

double HLTPerformanceInfo::totalCPUTime() const {
  double t = 0;
  for ( size_t i = 0; i < modules_.size(); ++i ) {
    t += modules_[i].cputime();
  }
  return t;
}

double HLTPerformanceInfo::totalPathTime(const size_t pathnumber)
{
  double t = 0;
  unsigned int cnt = 0;
  ModulesInPath::const_iterator i = paths_[pathnumber].begin();
  for ( ; i != paths_[pathnumber].end(); ++i ) {
    if ( cnt > paths_[pathnumber].status().index()) break;
    ++cnt;
    t += modules_[*i].time();
  }
  return t;
}

double HLTPerformanceInfo::totalPathCPUTime(const size_t pathnumber)
{
  double t = 0;
  unsigned int cnt = 0;
  ModulesInPath::const_iterator i = paths_[pathnumber].begin();
  for ( ; i != paths_[pathnumber].end(); ++i ) {
    if ( cnt > paths_[pathnumber].status().index()) break;
    ++cnt;
    t += modules_[*i].cputime();
  }
  return t;
}



double HLTPerformanceInfo::longestModuleTime() const {
  double t = -1;
  for ( Modules::const_iterator i = modules_.begin();
	i != modules_.end(); ++i ) {
    t = std::max(i->time(),t);
  }
  return t;
}

double HLTPerformanceInfo::longestModuleCPUTime() const {
  double t = -1;
  for ( Modules::const_iterator i = modules_.begin();
	i != modules_.end(); ++i ) {
    t = std::max(i->cputime(),t);
  }
  return t;
}

const char* HLTPerformanceInfo::longestModuleTimeName() const 
{
  double t = -1;
  std::string slowpoke("unknown");
  for ( Modules::const_iterator i = modules_.begin();
	i != modules_.end(); ++i ) {
    if ( i->time() > t ) {
      slowpoke = i->name();
      t = i->time();
    }
  }
  return slowpoke.c_str();
}
    
const char* HLTPerformanceInfo::longestModuleCPUTimeName() const 
{
  double t = -1;
  std::string slowpoke("unknown");
  for ( Modules::const_iterator i = modules_.begin();
	i != modules_.end(); ++i ) {
    if ( i->cputime() > t ) {
      slowpoke = i->name();
      t = i->cputime();
    }
  }
  return slowpoke.c_str();
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

HLTPerformanceInfo::Module & HLTPerformanceInfo::getModuleOnPath(size_t m, 
								 size_t p)
{
  // well if this doesn't get your attention....
  assert(p<paths_.size()&& m<paths_[p].numberOfModules());
  size_t j = paths_[p].getModuleIndex(m);
  return modules_.at(j);
}


bool HLTPerformanceInfo::uniqueModule(const char *mod) const {
  int mCtr = 0 ;
  for ( size_t p = 0; p < paths_.size(); ++p ) {
    for ( size_t m = 0; m < paths_[p].numberOfModules(); ++m ) {
      size_t modIndex = paths_[p].getModuleIndex(m);
      if ( modules_[modIndex].name() == std::string(mod) ) 
	++mCtr;
      if ( mCtr > 1 ) 
	return false;
    }
  }

  if (mCtr == 0) return false ;
  return true ;
}
