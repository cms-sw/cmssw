#ifndef PhysicsTools_Utilities_RootMinuitCommands_h
#define PhysicsTools_Utilities_RootMinuitCommands_h
#include "PhysicsTools/Utilities/interface/RootMinuit.h"
#include "PhysicsTools/Utilities/interface/ParameterMap.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "Utilities/General/interface/FileInPath.h"
#include "PhysicsTools/Utilities/interface/Parameter.h"
#include <map>
#include <string>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include<boost/tokenizer.hpp>

const char * kParameter = "par";
const char * kMinimize = "minimize";
const char * kMigrad = "migrad";

namespace fit {
  template<class Function>
  class RootMinuitCommands {
  public:
    typedef RootMinuit<Function> minuit;
    RootMinuitCommands(bool verbose = true) :
      verbose_(verbose) {
    }
    RootMinuitCommands(const char * fileName, bool verbose = true) :
      verbose_(verbose) {
      init(fileName);
    }
    void init(const char * fileName) {
      using namespace std;
      string cmssw_release_base = getenv("CMSSW_RELEASE_BASE");
      string cmssw_base = getenv("CMSSW_BASE");
      string path = "."; 
      if(!cmssw_release_base.empty()) {
	path += ':';
	path += (cmssw_release_base + "/src");
      }
      if(!cmssw_base.empty()) {
	path += ':';
	path += (cmssw_base + "/src");
      }
      FileInPath fileInPath(path, fileName);
      ifstream * file = fileInPath();
      if(file==0 || !file->is_open())
	throw edm::Exception(edm::errors::Configuration)
	  << "RootMinuitCommands: can't open file: " << fileName 
	  << " in path: " << path << "\n";
      if (verbose_) 
	cout << ">>> configuration file: " << fileName << endl;
      string line;
      while(getline(*file, line)) {
	line.erase(line.end()-1);
	using namespace boost;
	typedef tokenizer<char_separator<char> > tokenizer;
	char_separator<char> sep(" ");
	tokenizer tokens(line, sep);
	tokenizer::iterator i = tokens.begin(), e = tokens.end();
	if(tokens.begin()==tokens.end()) continue;
	if(*(i->begin()) != '#') {
	  if(*i == kParameter) {
	    string name = *(++i);
	    parameter_t par;
	    par.val = string2double(*(++i));
	    par.err = string2double(*(++i));
	    par.min = string2double(*(++i));
	    par.max = string2double(*(++i));
	    string fixed = *(++i);
	    if(fixed == "fixed") 
	      par.fixed = true;
	    else if(fixed == "free")
	      par.fixed = false;
	    else
	      throw edm::Exception(edm::errors::Configuration)
		<< "RootMinuitCommands: fix parameter option unknown: " << *i << "\n"
		<< "valid options are: fixed, free.\n";
	    pars_.push_back(std::make_pair(name, par));
	    size_t s = parIndices_.size();
	    parIndices_[name] = s;
	    if(verbose_)
	      cout << ">>> " << kParameter << " " << name 
		   << " " << par.val 
		   << " [" << par.min << ", " << par.max << "],"
		   << " err: " << par.err
		   << endl;
	  } else if(*i == kMinimize) {
	    command com;
	    com.name = kMinimize;
	    commands_.push_back(com);
	    if(verbose_)
	      cout << ">>> " << com.name << endl;
	  } else if(*i == kMigrad) {
	    command com;
	    com.name = kMigrad;
	    commands_.push_back(com);
	    if(verbose_)
	      cout << ">>> " << com.name << endl;
	  } else {
	    throw edm::Exception(edm::errors::Configuration)
	      << "RootMinuitCommands: unkonwn command:: " << *i
	      << "\n";
	    
	  }
	}
      }
      if (verbose_) 
	cout << ">>> end configuration" << endl;
    }
    double par(const std::string& name) {
      return parameter(name).val;
    }
    double err(const std::string& name) {
      return parameter(name).err;
    }  
    double min(const std::string& name) {
      return parameter(name).min;
    }
    double max(const std::string& name) {
      return parameter(name).max;
    }  
    bool fixed(const std::string& name) {
      return parameter(name).fixed;
    }   
    void add(RootMinuit<Function>& minuit, function::Parameter& p) const {
      const std::string & name = p.name();
      const parameter_t & par = parameter(name);
      minuit.addParameter(p, par.err, par.min, par.max);
      if(par.fixed) minuit.fixParameter(name);
    }
    void run(RootMinuit<Function>& minuit) const {
      using namespace std;
      typename vector<command>::const_iterator c = commands_.begin(), end = commands_.end();
      for(; c != end; ++c) {
	if(verbose_) {
	  cout << ">>> minuit command: ";
	  c->print(cout);
	  cout << endl;
	}
	if(c->name == kMinimize)
	  minuit.minimize();
	if(c->name == kMigrad)
	  minuit.migrad();
      }
    }
  private:
    bool verbose_;
    parameterVector_t pars_;
    std::map<std::string, size_t> parIndices_;
    struct command {
      std::string name;
      void print(std::ostream& cout) const {
	cout << name;
      }
    };
    std::vector<command> commands_;
    double string2double(const std::string & str) const {
      const char * begin = str.c_str();
      char * end;
      double val = strtod(begin, &end);
      size_t s = end - begin;
      if(s < str.size()) {
	throw edm::Exception(edm::errors::Configuration)
	  << "RootMinuitCommands: invalid double value: " 
	  << str  << "\n";
      }
      return val;
    }
    const parameter_t & parameter(const std::string& name) const {
      typename std::map<std::string, size_t>::const_iterator p = parIndices_.find(name);
      if(p == parIndices_.end())
	throw edm::Exception(edm::errors::Configuration)
	  << "RootMinuit: can't find parameter " << name << "\n";
      return pars_[p->second].second;
    }
  };
}

#endif
