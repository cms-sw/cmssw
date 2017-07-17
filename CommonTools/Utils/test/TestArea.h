#include<string>
#include<iostream>
#include <cstdlib>
#include <cassert>

struct TestArea {

   static void execIt(std::string const & command) {
     std::cerr <<"executing: " << command << std::endl;
     system(command.c_str());
   }

   explicit TestArea(std::string  const & pkg) {
     assert(!pkg.empty());
     std::cerr << "setting up " << pkg + "/ExprEval" << std::endl;
     srcArea += "$CMSSW_BASE/src/" +pkg;
     incArea += "$CMSSW_BASE/include/$SCRAM_ARCH/" +pkg;  

     // clean up (in case of previous failure)
     execIt(std::string("rm -rf ") + srcArea +" "+ incArea);

     execIt("echo CMSSW_BASE = $CMSSW_BASE");
     execIt(std::string("mkdir -p ") + srcArea + "/ExprEval/src");
     execIt(std::string("cp $CMSSW_BASE/src/CommonTools/Utils/test/ExprEvalStubs/*.h ") + srcArea + "/ExprEval/src/.");
     execIt(std::string("cp $CMSSW_BASE/src/CommonTools/Utils/test/ExprEvalStubs/BuildFile.xml ") + srcArea + "/ExprEval/.");
     execIt("pushd $CMSSW_BASE; scram b -j 2; popd");
     execIt("ls $CMSSW_BASE/src");
     execIt(std::string("ls -l ") + srcArea + "/ExprEval/src");
     execIt(std::string("ls -l ") + incArea +"/ExprEval/src");
  }

  ~TestArea() {
     execIt(std::string("rm -rf ") + srcArea +" "+ incArea);
  }
     std::string srcArea;
     std::string incArea;

};
