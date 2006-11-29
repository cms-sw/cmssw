// $Id$
#include <functional>
#include <boost/lambda/lambda.hpp> 
#include <boost/lambda/bind.hpp> 
#include <boost/bind.hpp> 
using namespace boost;

#include "DataFormats/HLTReco/interface/HLTPerformanceInfo.h"

HLTPerformanceInfo::HLTPerformanceInfo()
{
  paths_.clear(); modules_.clear();
}

double HLTPerformanceInfo::Path::time() const
{
  double t = 0;
  // old and busted
  for (ModulesInPath::const_iterator i = moduleView_.begin();
       i != moduleView_.end(); ++i ) {
    t += (*allModules_)[*i].time();
  }

  // new hotness
  for ( HLTPerformanceInfo::Path::const_iterator j = this->begin();
	j != this->end(); ++j ) {
    ;
    t += j->time();
  }

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
    std::cout << "addModuleToPath: Can't find Module " << mod 
	      << std::endl;
    Module newMod(mod, 0); // time = 0 since it wasn't run
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


double HLTPerformanceInfo::totalTime() const
{
  double t = 0;
  // if I were smarter I could use boost or something to do something like 
  // below. but I'm not.
//   double a = 
//     std::for_each(beginModules(), endModules(), 
// 		  bind(std::plus<double>(), 
// 		       bind(&Module::Time, _1),
// 		       t)
// 		  );
  // t = _1 + std::mem_fun_ref(&Module::Time);
  for ( Modules::const_iterator i = beginModules();
        i != endModules(); ++i ) {
    t += i->time();
  }
  return t;
}

