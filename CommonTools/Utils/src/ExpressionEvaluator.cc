#include "CommonTools/Utils/interface/ExpressionEvaluator.h"
#include "FWCore/Version/interface/GetReleaseVersion.h"
#include "FWCore/Utilities/interface/GetEnvironmentVariable.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "popenCPP.h"

#include <fstream>
#include <regex>
#include <dlfcn.h>

// #define VI_DEBUG

#ifdef VI_DEBUG
#include <iostream>
#define COUT std::cout
#else
#define COUT LogDebug("ExpressionEvaluator")
#endif

using namespace	reco::exprEvalDetails;


namespace {
  std::string generateName() {
    auto n1 = execSysCommand("uuidgen | sed 's/-//g'");
    n1.pop_back();
    return n1;
  }

 void remove(std::string const & name) {
  std::string sfile = "/tmp/"+name+".cc";
  std::string ofile = "/tmp/"+name+".so";

  std::string rm="rm -f "; rm+=sfile+' '+ofile;

  system(rm.c_str());

 }

 std::string patchArea() {
    auto n1 = execSysCommand("pushd $CMSSW_BASE > /dev/null;scram tool tag cmssw CMSSW_BASE; popd > /dev/null");
    n1.pop_back();
    COUT << "base area " << n1 << std::endl;
    return n1[0]=='/' ? n1 : std::string();
 }


}

namespace reco{

ExpressionEvaluator::ExpressionEvaluator(const char * pkg, const char * iname, std::string const & iexpr) :
  m_name("VI_"+generateName())
{


  std::string pch = pkg; pch += "/src/precompile.h";
  std::string quote("\"");

  
  std::string sfile = "/tmp/"+m_name+".cc";
  std::string ofile = "/tmp/"+m_name+".so";

  auto arch = edm::getEnvironmentVariable("SCRAM_ARCH");
  auto baseDir = edm::getEnvironmentVariable("CMSSW_BASE");
  auto relDir = edm::getEnvironmentVariable("CMSSW_RELEASE_BASE");


  std::string incDir = "/include/" + arch + "/";
  std::string cxxf;
  {
    // look in local dir
    std::string file = baseDir + incDir + pch + ".cxxflags";
    std::ifstream ss(file.c_str());
    COUT << "local file: " << file << std::endl;
    if (ss) {
      std::getline(ss,cxxf);
      incDir = baseDir + incDir;
    } else {
       // look in release area 
       std::string file = relDir + incDir + pch + ".cxxflags";
       COUT << "file in release area: " << file << std::endl;
       std::ifstream ss(file.c_str());
       if (ss) {
         std::getline(ss,cxxf);
         incDir = relDir + incDir;
       } else {
         // look in release is a patch area 
         auto paDir = patchArea();
         if (paDir.empty())  throw  cms::Exception("ExpressionEvaluator", "error in opening patch area for "+ baseDir);
         std::string file = paDir + incDir + pch + ".cxxflags";
         COUT << "file in base release area: " << file << std::endl;
         std::ifstream ss(file.c_str());
         if (!ss)  throw  cms::Exception("ExpressionEvaluator", pch + " file not found neither in " + baseDir + " nor in " + relDir  + " nor in " + paDir);
         std::getline(ss,cxxf);
         incDir = paDir + incDir;
       }
    }

    { std::regex rq("-I[^ ]+"); cxxf = std::regex_replace(cxxf,rq,std::string("")); }
    { std::regex rq("=\""); cxxf = std::regex_replace(cxxf,rq,std::string("='\"")); }
    { std::regex rq("\" "); cxxf = std::regex_replace(cxxf,rq,std::string("\"' ")); }
    COUT << '|' << cxxf << "|\n" << std::endl;

  }

  std::string cpp = "c++ -H -Wall -shared -Winvalid-pch "; cpp+=cxxf;
  cpp += " -I" + incDir; 
  cpp += " -o " + ofile + ' ' + sfile+" 2>&1\n";

  COUT << cpp << std::endl;


  //  prepare the file to compile
  std::string factory = "factory" + m_name;

  std::string source = std::string("#include ")+quote+ pch +quote+"\n";
  source+="struct "+m_name+" final : public "+iname + "{\n";
  source+=iexpr;
  source+="\n};\n";


  source += "extern " + quote+'C'+quote+' ' + std::string(iname) + "* "+factory+"() {\n";
  source += "static "+m_name+" local;\n";
  source += "return &local;\n}\n";


  COUT << source << std::endl;

  {
    std::ofstream tmp(sfile.c_str());
    tmp<<source << std::endl;                                                                        
  }



  // compile 
  auto ss = execSysCommand(cpp);
  COUT << ss << std::endl;

  void * dl = dlopen(ofile.c_str(),RTLD_LAZY);
  if (!dl) {
     remove(m_name);
     throw  cms::Exception("ExpressionEvaluator", std::string("compilation/linking failed\n") +  cpp + ss + "dlerror " + dlerror());
    return;
  }

  m_expr = dlsym(dl,factory.c_str());
  remove(m_name);
}


ExpressionEvaluator::~ExpressionEvaluator(){
  remove(m_name);
}



}
