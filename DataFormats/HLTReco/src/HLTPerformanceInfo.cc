// $Id: HLTPerformanceInfo.cc,v 1.7 2007/04/04 00:36:08 bdahmes Exp $
#include <functional>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <boost/lambda/lambda.hpp> 
#include <boost/lambda/bind.hpp> 
#include <boost/bind.hpp> 

#include "DataFormats/Common/interface/HLTenums.h"
#include "DataFormats/HLTReco/interface/HLTPerformanceInfo.h"


// THIS IS MY BABY!
// iterator adaptor to allow you to do some operation to 
// the return value of a member function.  intended for
// things like std::accumulate
template<class V, class Res, class binary_op>
class BinaryOpMemFun {
 private:
  // pointer to a member funtion of V which has no arguments and returns
  // Res.
  Res (V::* _s)() const;
 public:
  BinaryOpMemFun(Res (V::*s)() const ) :
    _s(s) {}
  // this is for for_each
  Res operator()( V & v) const {
    // this is the function call
    return (v.*_s)();
  }
  // this is for accumulate
  Res operator()(Res b,const  V & v) const {
    // it applies binary_op(b, v->_s()) and returns the result
    // e.g. if you use plus it's b+ v->_s()
    return binary_op()((v.*_s)(), b);
  }
};


HLTPerformanceInfo::HLTPerformanceInfo()
{
  paths_.clear(); modules_.clear();
}

double HLTPerformanceInfo::Path::time() const
{
  double t = 0;
  // we only want to add those up to the last one run.
  HLTPerformanceInfo::Path::const_iterator iter ;
  for (iter=this->begin(); iter!=this->end(); iter++) 
      if (iter->indexInPath(*this) <= int(this->status().index())) t += iter->time(); 

  return t;
}

double HLTPerformanceInfo::Path::cputime() const
{
  double t = 0;
  // we only want to add those up to the last one run.
  HLTPerformanceInfo::Path::const_iterator iter ;
  for (iter=this->begin(); iter!=this->end(); iter++) 
      if (iter->indexInPath(*this) <= int(this->status().index())) t += iter->cputime(); 

  return t;
}

void HLTPerformanceInfo::addModuleToPath(const char *mod, Path *p ) 
{
  Modules::const_iterator m = this->findModule(mod);

  if ( m != modules_.end() ) {
    size_t a = m - modules_.begin();
    p->addModuleRef(a);
  }
  else {
    // if we can't find the module, it probably just wasn't run. 
    // so no worries.
    Module newMod(mod, 0, 0); // time (wall and cpu) = 0 since it wasn't run
    modules_.push_back(newMod);
    p->addModuleRef(modules_.size()-1); // last guy on the stack
  }
}

void HLTPerformanceInfo::addPath(Path & p )
{
  // need this to get back at the modules that we don't own
  p.setModules_(&modules_);
    
  paths_.push_back(p);
}

void HLTPerformanceInfo::Module::setStatusByPath(Path* path)
{
  //--- Based on path status, define module status ---//
  unsigned int ctr = 0 ; 
  for ( HLTPerformanceInfo::Path::const_iterator iter = path->begin();
	iter!=path->end(); iter++ ) {
    edm::hlt::HLTState modState = edm::hlt::Ready ; 
    unsigned int modIndex = 0 ; 

    if (path->status().accept()) {
      modState = edm::hlt::Pass ;
    } else {
      if ( path->status().index() > ctr ) {
	modState = edm::hlt::Pass ; 
      } else if ( path->status().index() == ctr ) {
	modState = path->status().state() ; 
      }
    }
    if (iter->name() == this->name())
      this->setStatus(edm::HLTPathStatus(modState,modIndex)) ; 
    ctr++ ; 
  }
}

double HLTPerformanceInfo::totalTime() const
{
  double t = 0;
  t = std::accumulate(beginModules(), endModules(), 0.,
		      BinaryOpMemFun<HLTPerformanceInfo::Module, double,
		      std::plus<double> >(&HLTPerformanceInfo::Module::time));
  return t;
}

double HLTPerformanceInfo::totalCPUTime() const
{
  double t = 0;
  t = std::accumulate(beginModules(), endModules(), 0.,
		      BinaryOpMemFun<HLTPerformanceInfo::Module, double,
		      std::plus<double> >(&HLTPerformanceInfo::Module::cputime));
  return t;
}

HLTPerformanceInfo::Modules::const_iterator 
HLTPerformanceInfo::findModule(const char* moduleInstanceName) 
{
  return std::find(modules_.begin(), modules_.end(),
		   moduleInstanceName);
}

HLTPerformanceInfo::PathList::const_iterator 
HLTPerformanceInfo::findPath(const char* pathName)
{
  PathList::const_iterator l = std::find(paths_.begin(), paths_.end(),
					 pathName);
  return l; 
}

int HLTPerformanceInfo::Module::indexInPath(Path path) const
{
  int ctr = 0 ; 
  for ( HLTPerformanceInfo::Path::const_iterator iter = path.begin();
	iter!=path.end(); iter++ ) {
    if (iter->name() == this->name()) return ctr ; 
    ctr++ ; 
  }
  //--- Module not found in path ---//
  return -1 ; 
}

double HLTPerformanceInfo::Path::lastModuleTime() const
{
  double prev_time = -1;
  for ( HLTPerformanceInfo::Path::const_iterator j = this->begin();
	j != this->end(); ++j ) {
    if ( j->status().wasrun() && !(j->status().accept()) )
      return prev_time;
    prev_time = j->time();
  }
  return -2; // no modules on the path
}

double HLTPerformanceInfo::Path::lastModuleCPUTime() const
{
  double prev_time = -1;
  for ( HLTPerformanceInfo::Path::const_iterator j = this->begin();
	j != this->end(); ++j ) {
    if ( j->status().wasrun() && !(j->status().accept()) )
      return prev_time;
    prev_time = j->cputime();
  }
  return -2; // no modules on the path
}

double HLTPerformanceInfo::longestModuleTime() const
{
  double t = -1;
  // not sure why this does not work - I guess cuz max isn't a functor?
//   t = std::accumulate(beginModules(), endModules(), -99, 
// 		      BinaryOpMemFun<HLTPerformanceInfo::Module, double,
// 		      &std::max >(&HLTPerformanceInfo::Module::time));
  for ( Modules::const_iterator i = beginModules();
        i != endModules(); ++i ) {
    t = std::max(i->time(),t);
  }
  return t;
}

double HLTPerformanceInfo::longestModuleCPUTime() const
{
  double t = -1;
  // not sure why this does not work - I guess cuz max isn't a functor?
//   t = std::accumulate(beginModules(), endModules(), -99, 
// 		      BinaryOpMemFun<HLTPerformanceInfo::Module, double,
// 		      &std::max >(&HLTPerformanceInfo::Module::time));
  for ( Modules::const_iterator i = beginModules();
        i != endModules(); ++i ) {
    t = std::max(i->cputime(),t);
  }
  return t;
}

const char* HLTPerformanceInfo::longestModuleTimeName() const
{
  double t = -1;
  std::string slowpoke("unknown");
  for ( Modules::const_iterator i = beginModules();
        i != endModules(); ++i ) {
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
  for ( Modules::const_iterator i = beginModules();
        i != endModules(); ++i ) {
    if ( i->cputime() > t ) {
      slowpoke = i->name();
      t = i->cputime();
    }
  }
  return slowpoke.c_str();
}


HLTPerformanceInfo::Path::const_iterator 
HLTPerformanceInfo::Path::lastModuleByStatus() const
{
  const_iterator a = this->begin();
  assert(status_.index()<moduleView_.size());
  a += status_.index();
  return a;
  
}


const char*
HLTPerformanceInfo::Path::lastModuleByStatusName() const
{
  const_iterator a = this->begin();
  assert(status_.index()<moduleView_.size());
  a += status_.index();
  return a->name().c_str();
  
}


// copy constructor. Need this for the pointer from the
// paths back to the module list.
HLTPerformanceInfo::HLTPerformanceInfo(const HLTPerformanceInfo & rhs )
{
  modules_ = rhs.modules_;
  paths_ = rhs.paths_;

  for (PathList::iterator a = paths_.begin(); a != paths_.end(); ++a ) {
    a->setModules_(&modules_);
  }
}

// assignment operator. Needed, for same reason as copy constructor.
HLTPerformanceInfo & HLTPerformanceInfo::operator=(const 
						   HLTPerformanceInfo & rhs )
{
  modules_ = rhs.modules_;
  paths_ = rhs.paths_;

  for (PathList::iterator a = paths_.begin(); a != paths_.end(); ++a ) {
    a->setModules_(&modules_);
  }
  return *this;
}

bool HLTPerformanceInfo::uniqueModule(const char *mod) const {
  int mCtr = 0 ;
  PathList::const_iterator pIter ;
  Path::const_iterator mIter ;
  for (pIter=paths_.begin(); pIter!=paths_.end(); pIter++) {
    for (mIter=pIter->begin(); mIter!=pIter->end(); mIter++) {
      if (mIter->name() == mod) mCtr++ ;
      if (mCtr > 1) return false ;
    }
  }
  if (mCtr == 0) return false ;
  return true ;
}
