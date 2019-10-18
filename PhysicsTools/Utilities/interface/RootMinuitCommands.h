#ifndef PhysicsTools_Utilities_RootMinuitCommands_h
#define PhysicsTools_Utilities_RootMinuitCommands_h
#include "PhysicsTools/Utilities/interface/RootMinuit.h"
#include "PhysicsTools/Utilities/interface/ParameterMap.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "PhysicsTools/Utilities/interface/Parameter.h"
#include <map>
#include <string>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <iostream>
#include <boost/tokenizer.hpp>
#include <boost/algorithm/string/join.hpp>

const char* kParameter = "par";
const char* kFix = "fix";
const char* kRelease = "release";
const char* kSet = "set";
const char* kMinimize = "minimize";
const char* kMigrad = "migrad";
const char* kPrintAll = "print_all";

namespace fit {

  struct RootMinuitCommand {
    std::string name;
    std::vector<std::string> stringArgs;
    std::vector<double> doubleArgs;
    void print(std::ostream& cout) const {
      cout << name;
      if (stringArgs.size() > 0) {
        for (size_t i = 0; i != stringArgs.size(); ++i) {
          if (i != 0)
            cout << ",";
          cout << " \"" << stringArgs[i] << "\"";
        }
      }
      if (doubleArgs.size() > 0) {
        for (size_t i = 0; i != doubleArgs.size(); ++i) {
          if (i != 0)
            cout << ",";
          cout << " " << doubleArgs[i];
        }
      }
    }
  };

  template <class Function>
  class RootMinuitCommands {
  public:
    typedef RootMinuit<Function> minuit;
    typedef RootMinuitCommand command;
    RootMinuitCommands(bool verbose = true) : verbose_(verbose) {}
    RootMinuitCommands(const char* fileName, bool verbose = true) : verbose_(verbose) { init(fileName); }
    void init(const char* fileName);
    double par(const std::string& name) { return parameter(name).val; }
    double err(const std::string& name) { return parameter(name).err; }
    double min(const std::string& name) { return parameter(name).min; }
    double max(const std::string& name) { return parameter(name).max; }
    bool fixed(const std::string& name) { return parameter(name).fixed; }
    void add(RootMinuit<Function>& minuit, funct::Parameter& p) const {
      const std::string& name = p.name();
      const parameter_t& par = parameter(name);
      minuit.addParameter(p, par.err, par.min, par.max);
      if (par.fixed)
        minuit.fixParameter(name);
    }
    void run(RootMinuit<Function>& minuit) const;

  private:
    typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
    bool verbose_;
    unsigned int lineNumber_;
    parameterVector_t pars_;
    std::map<std::string, size_t> parIndices_;
    std::vector<command> commands_;
    double string2double(const std::string& str) const {
      const char* begin = str.c_str();
      char* end;
      double val = strtod(begin, &end);
      size_t s = end - begin;
      if (s < str.size()) {
        throw edm::Exception(edm::errors::Configuration) << "RootMinuitCommands: invalid double value: " << str << "\n";
      }
      return val;
    }
    const parameter_t& parameter(const std::string& name) const {
      typename std::map<std::string, size_t>::const_iterator p = parIndices_.find(name);
      if (p == parIndices_.end())
        throw edm::Exception(edm::errors::Configuration) << "RootMinuit: can't find parameter " << name << "\n";
      return pars_[p->second].second;
    }
    std::string errorHeader() const {
      std::ostringstream out;
      out << "RootMinuitCommands config. error, line " << lineNumber_ << ": ";
      return out.str();
    }
    std::string nextToken(typename tokenizer::iterator& i, const typename tokenizer::iterator& end) const {
      ++i;
      if (i == end)
        throw edm::Exception(edm::errors::Configuration) << errorHeader() << "missing parameter\n";
      return *i;
    }
  };

  template <typename Function>
  void RootMinuitCommands<Function>::init(const char* fileName) {
    using namespace std;
    string cmssw_release_base = std::getenv("CMSSW_RELEASE_BASE");
    string cmssw_base = std::getenv("CMSSW_BASE");
    vector<string> directories;
    directories.reserve(3);
    directories.emplace_back(".");
    if (!cmssw_release_base.empty()) {
      directories.emplace_back(cmssw_release_base + "/src");
    }
    if (!cmssw_base.empty()) {
      directories.emplace_back(cmssw_base + "/src");
    }
    ifstream file;
    for (auto const& d : directories) {
      std::ifstream f{d + "/" + fileName};
      if (f.good()) {
        file = std::move(f);
        break;
      }
    }
    if (!file.is_open()) {
      throw edm::Exception(edm::errors::Configuration)
          << "RootMinuitCommands: can't open file: " << fileName
          << " in path: " << boost::algorithm::join(directories, ":") << "\n";
    }
    if (verbose_)
      cout << ">>> configuration file: " << fileName << endl;
    string line;
    lineNumber_ = 0;
    bool commands = false;
    while (getline(file, line)) {
      ++lineNumber_;
      if (line.size() == 0)
        continue;
      char last = *line.rbegin();
      if (!(last >= '0' && last <= 'z'))
        line.erase(line.end() - 1);
      boost::char_separator<char> sep(" ");
      tokenizer tokens(line, sep);
      tokenizer::iterator i = tokens.begin(), e = tokens.end();
      if (tokens.begin() == tokens.end())
        continue;
      if (*(i->begin()) != '#') {
        if (*i == kParameter) {
          if (commands)
            throw edm::Exception(edm::errors::Configuration)
                << errorHeader() << "please, declare all parameter before all other minuit commands.\n";
          string name = nextToken(i, e);
          parameter_t par;
          par.val = string2double(nextToken(i, e));
          par.err = string2double(nextToken(i, e));
          par.min = string2double(nextToken(i, e));
          par.max = string2double(nextToken(i, e));
          tokenizer::iterator j = i;
          ++j;
          if (j != e) {
            string fixed = nextToken(i, e);
            if (fixed == "fixed")
              par.fixed = true;
            else if (fixed == "free")
              par.fixed = false;
            else
              throw edm::Exception(edm::errors::Configuration)
                  << errorHeader() << "fix parameter option unknown: " << *i << "\n"
                  << "valid options are: fixed, free.\n";
          } else {
            par.fixed = false;
          }
          pars_.push_back(std::make_pair(name, par));
          size_t s = parIndices_.size();
          parIndices_[name] = s;
          if (verbose_)
            cout << ">>> " << kParameter << " " << name << " " << par.val << " [" << par.min << ", " << par.max << "],"
                 << " err: " << par.err << endl;
        } else if (*i == kFix || *i == kRelease) {
          commands = true;
          command com;
          com.name = *i;
          string arg = nextToken(i, e);
          com.stringArgs.push_back(arg);
          commands_.push_back(com);
          if (verbose_) {
            cout << ">>> ";
            com.print(cout);
            cout << endl;
          }
        } else if (*i == kSet) {
          commands = true;
          command com;
          com.name = *i;
          string arg = nextToken(i, e);
          com.stringArgs.push_back(arg);
          com.doubleArgs.push_back(string2double(nextToken(i, e)));
          commands_.push_back(com);
          if (verbose_) {
            cout << ">>> ";
            com.print(cout);
            cout << endl;
          }
        } else if (*i == kMinimize || *i == kMigrad || *i == kPrintAll) {
          commands = true;
          command com;
          com.name = *i;
          commands_.push_back(com);
          if (verbose_) {
            cout << ">>> ";
            com.print(cout);
            cout << endl;
          }
        } else {
          throw edm::Exception(edm::errors::Configuration) << errorHeader() << "unkonwn command:: " << *i << "\n";
        }
      }
    }
    if (verbose_)
      cout << ">>> end configuration" << endl;
  }

  template <typename Function>
  void RootMinuitCommands<Function>::run(RootMinuit<Function>& minuit) const {
    using namespace std;
    typename vector<command>::const_iterator c = commands_.begin(), end = commands_.end();
    for (; c != end; ++c) {
      if (verbose_) {
        cout << ">>> minuit command: ";
        c->print(cout);
        cout << endl;
      }
      if (c->name == kMinimize)
        minuit.minimize();
      else if (c->name == kMigrad)
        minuit.migrad();
      else if (c->name == kPrintAll)
        minuit.printFitResults();
      else if (c->name == kFix)
        minuit.fixParameter(c->stringArgs[0]);
      else if (c->name == kRelease)
        minuit.releaseParameter(c->stringArgs[0]);
      else if (c->name == kSet)
        minuit.setParameter(c->stringArgs[0], c->doubleArgs[0]);
    }
  }

}  // namespace fit

#endif
