// -*-c++-*-
// $Id: HLTPerformanceInfo.h,v 1.15 2013/04/02 09:10:28 fwyzard Exp $
#ifndef HLTPERFORMANCEINFO_H
#define HLTPERFORMANCEINFO_H

#include <string>
#include <vector>

#include "FWCore/Utilities/interface/typedefs.h"
#include "DataFormats/Common/interface/HLTPathStatus.h"

class HLTPerformanceInfo {
public:
  class Path;
  class Module;
  typedef std::vector<Path>         PathList;
  typedef std::vector<Module>       Modules;
  typedef std::vector<cms_uint32_t> ModulesInPath;
  HLTPerformanceInfo();
  //
  class Module {
  private:
    std::string name_;  // module instance name
    double dt_;         // Wall-clock time
    double dtCPU_ ;     // CPU time
    // I am using this even for modules....
    edm::HLTPathStatus status_;
  public:
    Module() 
      : name_("unknown")
    {}
    // new constructor adding cpu time
    Module(const char *n, const double dt, const double dtCPU, 
	   edm::HLTPathStatus stat = edm::hlt::Ready)
      : name_(n), dt_(dt), dtCPU_(dtCPU), status_(stat)
    { }
    std::string name() const {
      return name_;
    }
    double time() const { return dt_; }
    double cputime() const { return dtCPU_; }
    edm::HLTPathStatus status() const { return status_; }
    bool operator==(const char *tname ) {
      return std::string(tname) == name();
    }
    void clear() {
      dt_ = 0 ;
      dtCPU_ = 0 ; 
      status_.reset();// = edm::hlt::Ready;
    }
    void setTime(double t) { dt_=t;}
    void setCPUTime(double t) { dtCPU_=t;}
    void setStatus(edm::HLTPathStatus status) { status_=status;} 
    // pw - can't a module be on different paths?
    //void setStatusByPath(Path *path) ; 
    //int indexInPath(Path path) const ; 
        
  };
  // end Module class definition

  // in this version the module can no longer iterate over the paths
  // by itself, since it has no access to the actual module list.
  class Path {
  private:
    std::string name_;
    ModulesInPath moduleView_; // indices into the module vector
    edm::HLTPathStatus status_;
        
  public:
    Path(const std::string n = "unknown") : 
      name_(n),
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
    
    const size_t operator[](size_t m) const {
      return moduleView_.at(m);
    }
    
    void addModuleRef( size_t m) {
      moduleView_.push_back(m);
    }
    
    ModulesInPath::const_iterator begin() {
      return moduleView_.begin();
    }
    ModulesInPath::const_iterator end() {
      return moduleView_.end();
    }
    size_t getModuleIndex(size_t j) const {
      return moduleView_.at(j);
    }
    
    size_t numberOfModules() const { return moduleView_.size(); };
    
  };
  // end Path class definition
private:
  PathList paths_;
  Modules modules_;

public:
  void addPath(const Path & p) {
    paths_.push_back(p);
  }
  void addModule(const Module & m ) {
    modules_.push_back(m);
  } 
  // by name
   void addModuleToPath(const char *mod, const char *path) {
     // first make sure module exists
     Modules::iterator m = findModule(mod);
     if ( m == endModules() ) {
       // new module - create it and stick it on the end
       Module newMod(mod, 0, 0); // time (wall and cpu) = 0 since it wasn't run	 
       modules_.push_back(newMod);
     }

     for ( size_t i = 0; i < paths_.size(); ++i ) {
       if ( !( paths_[i] == path ) ) continue;
       // we found the path, add module to the end
       for ( size_t j = 0; j < modules_.size(); ++j ) {
	 if ( !(modules_[j] == mod) ) continue;
	 paths_[i].addModuleRef(j);
	 break;
       }
       break;
     }
   }
  // by index
  void addModuleToPath(const size_t mod, const size_t path) {
    assert(( path <paths_.size()) && (mod < modules_.size()) );
    paths_[path].addModuleRef(mod);
  }

  void clear() {
    modules_.clear(); paths_.clear();
  }
  void clearModules() {
    for ( size_t i = 0; i < modules_.size(); ++i ) {
      modules_[i].clear();
    }
  }

  // non-const?
  const Module & getModuleOnPath(size_t m, size_t p) const ;
  const Module & getModule(size_t m) const { return modules_.at(m); }
  const Path & getPath(size_t p) const { return paths_.at(p); }


  // find a module, given its name.
  // returns endModules() on failure
  Modules::iterator findModule(const char* moduleInstanceName) ;
  PathList::iterator findPath(const char* pathName) ;

  int moduleIndexInPath(const char *mod, const char *path);

  size_t numberOfPaths() const {
    return paths_.size();
  }
  size_t numberOfModules() const {
    return modules_.size();
  }

  PathList::iterator beginPaths()  {
       return paths_.begin();
  }
  PathList::iterator endPaths()  {
    return paths_.end();
  }
    
  Modules::const_iterator beginModules() const {
    return modules_.begin();
  }
  
  Modules::const_iterator endModules() const {
    return modules_.end();
  }
    
  double totalTime() const;
  double totalCPUTime() const;
  double longestModuleTime() const;
  double longestModuleCPUTime() const;
  const char* longestModuleTimeName() const;
  const char* longestModuleCPUTimeName() const;

  double totalPathTime(const size_t path);
  double totalPathCPUTime(const size_t path);

  
  void setStatusOfModulesFromPath(const char* pathName);
    
//   double lastModuleOnPathTime(const size_t pathnumber);
//   double lastModuleOnPathCPUTime(const size_t pathnumber);
  
  // is this module only on one path?
  bool uniqueModule(const char *mod) const ;
};


typedef std::vector<HLTPerformanceInfo> HLTPerformanceInfoCollection;

#endif // HLTPERFORMANCEINFO_H
