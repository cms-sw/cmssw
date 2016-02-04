#include <stdlib.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <algorithm>
#include <string>

#include "DataFormats/Common/interface/Timestamp.h"
#include "CondCore/IOVService/interface/IOV.h"
#include "CondTools/Hcal/interface/HcalDbTool.h"

#include "CondFormats/HcalObjects/interface/HcalPedestals.h"
#include "CondFormats/HcalObjects/interface/HcalPedestalWidths.h"
#include "CondFormats/HcalObjects/interface/HcalGains.h"
#include "CondFormats/HcalObjects/interface/HcalGainWidths.h"
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/HcalObjects/interface/HcalQIEData.h"
#include "CondFormats/HcalObjects/interface/HcalCalibrationQIEData.h"


namespace {
  typedef HcalDbTool::IOVRun IOVRun;
  typedef std::map<IOVRun,std::string> IOVCollection;
  typedef std::pair<IOVRun,IOVRun> IntervalOV;

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
  std::cout << "Initialize POOL DB with dummy HCAL data" << std::endl;
  std::cout << "    feedback -> ratnikov@fnal.gov" << std::endl;
  std::cout << "Use:" << std::endl;
  sprintf (buffer, " %s <parameters>\n", args.command ().c_str());
  std::cout << buffer;
  args.printOptionsHelp ();
}

void initDb (const std::string& inputdb, bool verbose) {
  HcalDbTool db (inputdb, verbose);
  if (inputdb.find ("sqlite") != std::string::npos) {
    unsigned pos = inputdb.find (':');
    if (pos != std::string::npos) {
      std::string filename (inputdb, pos+1);
      std::cout << "initDb-> creating file " << filename << std::endl;
      std::ifstream file (filename.c_str()); // touch file
      file.close ();
    }
  }
  // make dummy metadata entry
  std::cout << "initDb-> initializing metadata" << std::endl;
  db.metadataSetTag ("dummy_tag", "dummy_token");
  // make dummy objects
  std::cout << "initDb-> initializing pedestals" << std::endl;
  HcalPedestals* peds = new HcalPedestals;
  db.putObject (peds, "dummy_pedestals", 1);
  std::cout << "initDb-> initializing pedestal widths" << std::endl;
  HcalPedestalWidths* pedws = new HcalPedestalWidths;
  db.putObject (pedws, "dummy_pedestalwidths", 1);
  std::cout << "initDb-> initializing gains" << std::endl;
  HcalGains* gains = new HcalGains;
  db.putObject (gains, "dummy_gains", 1);
  std::cout << "initDb-> initializing gain widths" << std::endl;
  HcalGainWidths* gainws = new HcalGainWidths;
  db.putObject (gainws, "dummy_gainwidths", 1);
  std::cout << "initDb-> initializing qie" << std::endl;
  HcalQIEData* qie = new HcalQIEData;
  db.putObject (qie, "dummy_qie", 1);
  std::cout << "initDb-> initializing electronics map" << std::endl;
  HcalElectronicsMap* map = new HcalElectronicsMap;
  db.putObject (map, "dummy_map", 1);
  std::cout << "initDb-> initializing channel quality" << std::endl;
  HcalChannelQuality* quality = new HcalChannelQuality;
  db.putObject (quality, "dummy_quality", 1);
}

} // namespace

int main (int argn, char* argv []) {

  Args args;
  args.defineParameter ("-db", "DB connection string, POOL format, i.e. oracle://devdb10/CMS_COND_HCAL");
  args.defineOption ("-help", "this help");
  args.defineOption ("-verbose", "verbose");
  
  args.parse (argn, argv);
  
  std::vector<std::string> arguments = args.arguments ();

  if (args.getParameter ("-db").empty() || args.optionIsSet ("-help")) {
    printHelp (args);
    return -1;
  }

  std::string inputdb = args.getParameter ("-db");
  bool verbose = args.optionIsSet ("-verbose");

  initDb (inputdb, verbose);
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
