#include <stdlib.h>
#include <vector>
#include <map>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>

#include "CondFormats/HcalObjects/interface/HcalPedestals.h"
#include "CondFormats/HcalObjects/interface/HcalPedestalWidths.h"
#include "CondTools/Hcal/interface/HcalDbOnline.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalPedestalAnalysis.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"

namespace {
  bool defaultsFile (const std::string fParam) {
    return fParam == "defaults";
  }

  bool asciiFile (const std::string fParam) {
    return fParam.find (':') == std::string::npos && std::string (fParam, fParam.length () - 4) == ".txt";
  }
  
  bool xmlFile (const std::string fParam) {
    return fParam.find (':') == std::string::npos && std::string (fParam, fParam.length () - 4) == ".xml";
  }
  
  bool dbFile (const std::string fParam) {
    return fParam.find (':') != std::string::npos;
  }

  bool masterDb (const std::string fParam) {
    return fParam.find ('@') != std::string::npos;
  }

  template <class T> 
  bool getObject (T* fObject, const std::string& fDb, const std::string& fTag, int fRun) {
    if (!fObject) return false;
    if (fDb.empty ()) return false; 
    if (asciiFile (fDb)) {
      std::ifstream stream (fDb.c_str ());
      HcalDbASCIIIO::getObject (stream, fObject); 
      return true;
    }
    else if (masterDb (fDb)) {
      std::cout << "HcalPedestalValidator-> Use input: MasterDB " << fDb << std::endl;
      HcalDbOnline masterDb (fDb);
      return masterDb.getObject (fObject, fTag, fRun);
    }
    else {
      return false;
    }
  }
  
  template <class T>
  bool putObject (T** fObject, const std::string& fDb, const std::string& fTag, int fRun) {
    if (!fObject || !*fObject) return false;
    if (fDb.empty ()) return false;
    if (asciiFile (fDb)) {
      std::ofstream stream (fDb.c_str ());
      HcalDbASCIIIO::dumpObject (stream, **fObject);
      return true;
    }
    else {
      return false;
    }
  }
}

// Args is a copy-paste from Fedor's peds_txt2xml.cc
class Args {
 public:
  Args () {};
  ~Args () {};
  void defineOption (const std::string& fOption, const std::string& fComment = "");
  void defineParameter (const std::string& fParameter, const std::string& fComment = "");
  void parse (int nArgs, char* fArgs []);
  void printOptionsHelp () const;
  std::string command () const;
  std::vector<std::string> arguments () const;
  bool optionIsSet (const std::string& fOption) const;
  std::string getParameter (const std::string& fKey);
 private:
  std::string mProgramName;
  std::vector <std::string> mOptions;
  std::vector <std::string> mParameters;
  std::vector <std::string> mArgs;
  std::map <std::string, std::string> mParsed;
  std::map <std::string, std::string> mComments;
};

int main (int argn, char* argv []) {

// CORAL required variables to be set, even if not needed
  const char* foo1 = "CORAL_AUTH_USER=blaaah";
  const char* foo2 = "CORAL_AUTH_PASSWORD=blaaah";
  if (!::getenv("CORAL_AUTH_USER")) ::putenv(const_cast<char*>(foo1));
  if (!::getenv("CORAL_AUTH_PASSWORD")) ::putenv(const_cast<char*>(foo2)); 

  Args args;
  args.defineParameter ("-p", "raw pedestals");
  args.defineParameter ("-w", "raw widths");
  args.defineParameter ("-run", "current run number <0>");
  args.defineParameter ("-ptag", "raw pedestal tag <NULL>");
  args.defineParameter ("-wtag", "raw width tag <ptag>");
  args.defineParameter ("-pref", "reference pedestals");
  args.defineParameter ("-wref", "reference widths");
  args.defineParameter ("-ptagref", "reference pedestal tag <NULL>");
  args.defineParameter ("-wtagref", "reference width tag <ptagref>");
  args.defineParameter ("-pval", "validated pedestals");
  args.defineParameter ("-wval", "validated widths");
  args.defineOption ("-help", "this help");

  args.parse (argn, argv);
  std::vector<std::string> arguments = args.arguments ();
  if (args.optionIsSet ("-help")) {
    args.printOptionsHelp ();
    return -1;
  }

// read parameters from command line
  std::string RawPedSource = args.getParameter("-p");
  std::string RawPedWidSource = args.getParameter("-w");
  std::string RawPedTag = args.getParameter("-ptag").empty() ? "" : args.getParameter("-ptag");
  std::string RawPedWidTag = args.getParameter("-wtag").empty() ? RawPedTag : args.getParameter("-wtag");
  int RawPedRun = args.getParameter("-run").empty() ? 0 : (int)strtoll (args.getParameter("-run").c_str(),0,0);
  int RawPedWidRun = RawPedRun;
  std::string RefPedSource = args.getParameter("-pref");
  std::string RefPedWidSource = args.getParameter("-wref");
  std::string RefPedTag = args.getParameter("-ptagref").empty() ? "" : args.getParameter("-ptagref");
  std::string RefPedWidTag = args.getParameter("-wtagref").empty() ? RefPedTag : args.getParameter("-wtagref");
  int RefPedRun = RawPedRun;
  int RefPedWidRun = RefPedRun;
  std::string outputPedDest = args.getParameter("-pval");
  std::string outputPedWidDest = args.getParameter("-wval");
  std::string outputPedTag = "";
  std::string outputPedWidTag = "";
  int outputPedRun = RawPedRun;
  int outputPedWidRun = outputPedRun;

  // need to know how to make proper topology in the future.
  HcalTopology topo(HcalTopologyMode::LHC,2,4);

// get reference objects
  HcalPedestals* RefPeds = 0;
  RefPeds = new HcalPedestals (&topo);
  if (!getObject (RefPeds, RefPedSource, RefPedTag, RefPedRun)) {
    std::cerr << "HcalPedestalValidator-> Failed to get reference Pedestals" << std::endl;
    return 1;
  }
  HcalPedestalWidths* RefPedWids = 0;
  RefPedWids = new HcalPedestalWidths (&topo);
  if (!getObject (RefPedWids, RefPedWidSource, RefPedWidTag, RefPedWidRun)) {
    std::cerr << "HcalPedestalValidator-> Failed to get reference PedestalWidths" << std::endl;
    return 2;
  }

// get input raw objects
  HcalPedestals* RawPeds = 0;
  RawPeds = new HcalPedestals (&topo);
  if (!getObject (RawPeds, RawPedSource, RawPedTag, RawPedRun)) {
    std::cerr << "HcalPedestalValidator-> Failed to get raw Pedestals" << std::endl;
    return 3;
  }
  HcalPedestalWidths* RawPedWids = 0;
  RawPedWids = new HcalPedestalWidths (&topo);
  if (!getObject (RawPedWids, RawPedWidSource, RawPedWidTag, RawPedWidRun)) {
    std::cerr << "HcalPedestalValidator-> Failed to get raw PedestalWidths" << std::endl;
    return 4;
  }

// make output objects
  HcalPedestals* outputPeds = 0;
  outputPeds = new HcalPedestals (&topo);
  HcalPedestalWidths* outputPedWids = 0;
  outputPedWids = new HcalPedestalWidths (&topo);

// run algorithm
  int nstat[4]={2500,2500,2500,2500};
  int Flag=HcalPedestalAnalysis::HcalPedVal(nstat,RefPeds,RefPedWids,RawPeds,RawPedWids,outputPeds,outputPedWids);

  delete RefPeds;
  delete RefPedWids;
  delete RawPeds;
  delete RawPedWids;


// store new objects if necessary
  if (Flag%100000>0) {
    if (outputPeds) {
      if (!putObject (&outputPeds, outputPedDest, outputPedTag, outputPedRun)) {
	  std::cerr << "HcalPedestalAnalyzer-> Failed to put output Pedestals" << std::endl;
          return 5;
	}
    }
    if (outputPedWids) {
	if (!putObject (&outputPedWids, outputPedWidDest, outputPedWidTag, outputPedWidRun)) {
        std::cerr << "HcalPedestalAnalyzer-> Failed to put output PedestalWidths" << std::endl;
        return 6;
      }
    }
  }
  delete outputPeds;
  delete outputPedWids;

return 0;
}

//==================== Args ===== BEGIN ==============================
void Args::defineOption (const std::string& fOption, const std::string& fComment) {
  mOptions.push_back (fOption);
  mComments [fOption] = fComment;
}

void Args::defineParameter (const std::string& fParameter, const std::string& fComment) {   mParameters.push_back (fParameter);
  mComments [fParameter] = fComment;
}

void Args::parse (int nArgs, char* fArgs []) {
  if (nArgs <= 0) return;
  mProgramName = std::string (fArgs [0]);
  int iarg = 0;
  while (++iarg < nArgs) {
    std::string arg (fArgs [iarg]);
    if (arg [0] != '-') mArgs.push_back (arg);
    else {
      if (std::find (mOptions.begin(), mOptions.end (), arg) !=  mOptions.end ()) {
        mParsed [arg] = "";
      }
      if (std::find (mParameters.begin(), mParameters.end (), arg) !=  mParameters.end ()) {
        if (iarg >= nArgs) {
          std::cerr << "ERROR: Parameter " << arg << " has no value specified. Ignore parameter." << std::endl;
        }
        else {
          mParsed [arg] = std::string (fArgs [++iarg]);
        }
      }
    }
  }
}

void Args::printOptionsHelp () const {
  char buffer [1024];
  std::cout << "Parameters:" << std::endl;
  for (unsigned i = 0; i < mParameters.size (); i++) {
    std::map<std::string, std::string>::const_iterator it = mComments.find (mParameters [i]);
    std::string comment = it != mComments.end () ? it->second : "uncommented";
    sprintf (buffer, "  %-8s <value> : %s", (mParameters [i]).c_str(),  comment.c_str());
    std::cout << buffer << std::endl;
  }
  std::cout << "Options:" << std::endl;
  for (unsigned i = 0; i < mOptions.size (); i++) {
    std::map<std::string, std::string>::const_iterator it = mComments.find (mOptions [i]);
    std::string comment = it != mComments.end () ? it->second : "uncommented";
    sprintf (buffer, "  %-8s  : %s", (mOptions [i]).c_str(),  comment.c_str());
    std::cout << buffer << std::endl;
  }
}

std::string Args::command () const {
  int ipos = mProgramName.rfind ('/');
  return std::string (mProgramName, ipos+1);
}

std::vector<std::string> Args::arguments () const {return mArgs;}

bool Args::optionIsSet (const std::string& fOption) const {
  return mParsed.find (fOption) != mParsed.end ();
}

std::string Args::getParameter (const std::string& fKey) {
  if (optionIsSet (fKey)) return mParsed [fKey];
  return "";
}

