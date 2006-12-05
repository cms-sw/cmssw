// -*-c++-*-
// $Id: HLTPerformanceInfo.h,v 1.2 2006/12/04 15:08:23 wittich Exp $
#ifndef HLTPERFORMANCEINFO_H
#define HLTPERFORMANCEINFO_H

#include <string>
#include <vector>

#include "boost/iterator/transform_iterator.hpp"

#include "DataFormats/Common/interface/HLTPathStatus.h"



class HLTPerformanceInfo
{
 public:
  class Path;
  class Module;
  typedef std::vector<Path> PathList;
  typedef std::vector<Module> Modules;
  typedef std::vector<size_t> ModulesInPath;
  HLTPerformanceInfo();
  // need to make a copy constructor due to 
  HLTPerformanceInfo(const HLTPerformanceInfo & rhs);
  HLTPerformanceInfo & operator=(const HLTPerformanceInfo & rhs);
  ~HLTPerformanceInfo() {}

  ///////////////////////////////////////////////////
  class Module {
  private:
    std::string name_; // module instance name
    double dt_;
    // I am using this even for modules....
    edm::HLTPathStatus status_;
  public:
  Module() 
    : name_("unknown")
    {}
  Module(const char *n, const double dt, 
	 edm::HLTPathStatus stat = edm::hlt::Ready)
    : name_(n), dt_(dt), status_(stat)
    { }
    std::string name() const {
      return name_;
    }
    double time() const { return dt_; }
    edm::HLTPathStatus status() const { return status_; }
    bool operator==(const char *tname ) {
      return std::string(tname) == name();
    }
    void clear() {
      dt_ = 0;
      status_.reset();// = edm::hlt::Ready;
    }
  };
  ///////////////////////////////////////////////////
  class Path {
  private:
    std::string name_;
    const Modules * allModules_; // need this
    ModulesInPath moduleView_; // does not own the modules
    edm::HLTPathStatus status_;

    void setModules_(const Modules *m ) {
      allModules_ = m;
    }           
    // need this friend declaration for containing to call setModules
    friend class HLTPerformanceInfo;
  public:
    // Adapter for iterator to make the mapping btw modules on 
    // paths and modules
    // thanks chris!
    class Adapter : public std::unary_function<size_t,const Module &> {
    private:
      const Modules *m_;
    public:
      Adapter(const Modules *a=0) : m_(a) { }
      const Module & operator()(size_t i) const {
	return (*m_)[i];
      }
    };

    typedef boost::transform_iterator<Adapter, ModulesInPath::const_iterator> 
    const_iterator;
    
    const_iterator begin() const {
      return const_iterator(moduleView_.begin(),Adapter(allModules_));
    }
    const_iterator end() const {
      return const_iterator(moduleView_.end(),Adapter(allModules_));
    }

    Path(const std::string n = "unknown") : 
      name_(n),
      allModules_(0),
      moduleView_(),
      status_()
    {}
    std::string name() const {
      return name_;
    }
    void setStatus( const edm::HLTPathStatus & result ) {
      status_ = result;
    }
    edm::HLTPathStatus status() const {
      return status_;
    }
    void clear() {
      status_.reset();
      // time is dynamic, nothing to reset
    }
    bool operator==( const char* tname) {
      return (std::string(tname) == name());
    }
    double time() const;

    double lastModuleTime() const;

    void addModuleRef( size_t m) {
      moduleView_.push_back(m);
    }

    const_iterator lastModuleByStatus() const;
    const char* lastModuleByStatusName() const;

    size_t numberOfModules() const { return moduleView_.size(); };
    
  };
  ///////////////////////////////////////////////////

 private:
  PathList paths_; // owns the paths
  Modules modules_; // owns the modules, since each can be on more than one path

 public:
  void addPath(Path & p ); 
  void addModule(const Module & m ) {
    modules_.push_back(m);
  }

  void clear() {
    modules_.clear(); paths_.clear();
  }

  // add a module pointed to the end of a path
  void addModuleToPath(const char *mod, Path *p );

  // find a module, given its name.
  // returns endModules() on failure
  Modules::const_iterator findModule(const char* moduleInstanceName) ;
  PathList::const_iterator findPath(const char* pathName) ;
 
  size_t numberOfPaths() const {
    return paths_.size();
  }
  size_t numberOfModules() const {
    return modules_.size();
  }

  PathList::const_iterator beginPaths() const {
    return paths_.begin();
  }
  PathList::const_iterator endPaths() const {
    return paths_.end();
  }

  Modules::const_iterator beginModules() const {
    return modules_.begin();
  }

  Modules::const_iterator endModules() const {
    return modules_.end();
  }

  double totalTime() const;
  double longestModuleTime() const;
  const char* longestModuleTimeName() const;
};

typedef std::vector<HLTPerformanceInfo> HLTPerformanceInfoCollection;

#endif // HLTPERFORMANCEINFO_H
