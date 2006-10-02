#include <stdlib.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <algorithm>
#include <string>

#include "CondCore/IOVService/interface/IOV.h"
#include "CondTools/Hcal/interface/HcalDbTool.h"
#include "CondTools/Hcal/interface/HcalDbOnline.h"

namespace {
  typedef HcalDbTool::IOVRun IOVRun;
  typedef std::map<IOVRun,std::string> IOVCollection;
  typedef std::pair<IOVRun,IOVRun> IntervalOV;

  std::vector <IntervalOV> allIOV (const cond::IOV& fIOV) {
    std::vector <IntervalOV> result;
    IOVRun iovMin = 0;
    for (IOVCollection::const_iterator iovi = fIOV.iov.begin (); iovi != fIOV.iov.end (); iovi++) {
      IOVRun iovMax = iovi->first;
      result.push_back (std::make_pair (iovMin, iovMax));
      iovMin = iovMax;
    }
    return result;
  }

bool dbFile (const std::string fParam) {
  return fParam.find (':') != std::string::npos;
}

bool onlineFile (const std::string fParam) {
  return fParam.find ('@') != std::string::npos &&
    fParam.find ("cms_val_lb") == std::string::npos;
}

}

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

void printHelp (const Args& args) {
  char buffer [1024];
  std::cout << "Tool to manipulate by IOV of Hcal Calibrations" << std::endl;
  std::cout << "    feedback -> ratnikov@fnal.gov" << std::endl;
  std::cout << "Use:" << std::endl;
  sprintf (buffer, " %s <what> <options> <parameters>\n", args.command ().c_str());
  std::cout << buffer;
  std::cout << "  where <what> is: \n    tag\n    iov" << std::endl;
  args.printOptionsHelp ();
}

void printTags (const std::string& fDb, bool fVerbose) {
  std::vector<std::string> allTags;
  if (dbFile (fDb)) {
    HcalDbTool poolDb (fDb, fVerbose);
    allTags = poolDb.metadataAllTags ();
    std::cout << "Tags available in CMS POOL DB instance: " << fDb << std::endl;
  }
  if (onlineFile (fDb)) {
    HcalDbOnline onlineDb (fDb, fVerbose);
    allTags = onlineDb.metadataAllTags ();
    std::cout << "Tags available in HCAL master DB instance: " << fDb << std::endl;
  }
  for (unsigned i = 0; i < allTags.size(); i++) {
    std::cout << allTags[i] << std::endl;
  }
}

void printRuns (const std::string& fDb, const std::string fTag, bool fVerbose) {
  std::vector <IntervalOV> allIOVs;
  if (dbFile (fDb)) {
    HcalDbTool poolDb (fDb, fVerbose);
    cond::IOV iov;
    if (poolDb.getObject (&iov, fTag)) {
      allIOVs = allIOV (iov);
      std::cout << "IOVs available for tag " << fTag << " in CMS POOL DB instance: " << fDb << std::endl;
    }
    else {
      std::cerr << "printRuns-> can not find IOV for tag " << fTag << std::endl;
    }
  }
  if (onlineFile (fDb)) {
    HcalDbOnline onlineDb (fDb, fVerbose);
    allIOVs = onlineDb.getIOVs (fTag);
  }
  for (unsigned i = 0; i < allIOVs.size(); i++) {
    std::cout << "[ " << allIOVs[i].first << " ... " << allIOVs[i].second << " )" << std::endl;
  }
}

int main (int argn, char* argv []) {

  Args args;
  args.defineParameter ("-db", "DB connection string, POOL format, or online format");
  args.defineParameter ("-tag", "tag specifyer");
  args.defineOption ("-help", "this help");
  args.defineOption ("-verbose", "this help");
  
  args.parse (argn, argv);
  
  std::vector<std::string> arguments = args.arguments ();

  if (arguments.size () < 1 || args.optionIsSet ("-help")) {
    printHelp (args);
    return -1;
  }

  std::string db = args.getParameter ("-db");
  std::string tag = args.getParameter ("-tag");
  bool verbose = args.optionIsSet ("-verbose");

  std::string what = arguments [0];

  if (what == "tag") {
    printTags (db, verbose);
  }
  else if (what == "iov") {
    printRuns (db, tag, verbose);
  }
  std::cout << "Program has completed successfully" << std::endl;
  return 0;
}


//==================== Args ===== BEGIN ==============================
void Args::defineOption (const std::string& fOption, const std::string& fComment) {
  mOptions.push_back (fOption);
  mComments [fOption] = fComment;
}

void Args::defineParameter (const std::string& fParameter, const std::string& fComment) {
  mParameters.push_back (fParameter);
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
//==================== Args ===== END ==============================
