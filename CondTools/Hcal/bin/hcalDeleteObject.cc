#include <stdlib.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <algorithm>
#include <string>

// other
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"
#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"


#include "CondCore/IOVService/interface/IOV.h"

// Hcal calibrations
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbHardcode.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbXml.h"
#include "CondTools/Hcal/interface/HcalDbOnline.h"
#include "CondTools/Hcal/interface/HcalDbTool.h"
#include "CondFormats/HcalObjects/interface/HcalPedestals.h"
#include "CondFormats/HcalObjects/interface/HcalPedestalWidths.h"
#include "CondFormats/HcalObjects/interface/HcalGains.h"
#include "CondFormats/HcalObjects/interface/HcalGainWidths.h"
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/HcalObjects/interface/HcalQIEData.h"
#include "CondFormats/HcalObjects/interface/HcalCalibrationQIEData.h"

//using namespace cms;


typedef HcalDbTool::IOVRun IOVRun;
typedef std::map<IOVRun,std::string> IOVCollection;


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
  std::cout << "Tool to delete object from Hcal Calibrations" << std::endl;
  std::cout << "CAUTION: Could cause inconsistency for running jobs using DB" << std::endl;
  std::cout << "    feedback -> ratnikov@fnal.gov" << std::endl;
  std::cout << "Use:" << std::endl;
  sprintf (buffer, " %s <what> <options> <parameters>\n", args.command ().c_str());
  std::cout << buffer;
  std::cout << "  where <what> is: \n    pedestals\n    gains\n    pwidths\n    gwidths\n    emap\n    qie\n    calibqie" << std::endl;
  args.printOptionsHelp ();
}

template <class T> bool deleteObject (T* fObject, 
				      const std::string& fInput, const std::string& fInputTag, HcalDbTool::IOVRun fInputRun, bool fVerbose) {
  HcalDbTool poolDb (fInput, fVerbose);
  return poolDb.deleteObject (fObject, fInputTag, fInputRun);
}

int main (int argn, char* argv []) {

  Args args;
  args.defineParameter ("-db", "DB connection string, POOL format");
  args.defineParameter ("-run", "run # for which constands should be deleted");
  args.defineParameter ("-tag", "tag for the input constants set");
  args.defineOption ("-verbose", "makes program verbose");
  
  args.parse (argn, argv);
  
  std::vector<std::string> arguments = args.arguments ();

  if (arguments.size () < 1 || args.optionIsSet ("-help")) {
    printHelp (args);
    return -1;
  }

  std::string input = args.getParameter ("-db");
  
  unsigned inputRun = args.getParameter ("-run").empty () ? 0 : strtoull (args.getParameter ("-run").c_str (), 0, 0);
  std::string inputTag = args.getParameter ("-tag");

  bool verbose = args.optionIsSet ("-verbose");


  std::string what = arguments [0];

  if (what == "pedestals") {
    HcalPedestals* object = 0;
    deleteObject (object, input, inputTag, inputRun, verbose);
  }
  else if (what == "gains") {
    HcalGains* object = 0;
  }
  else if (what == "pwidths") {
    HcalPedestalWidths* object = 0;
  }
  else if (what == "gwidths") {
    HcalGainWidths* object = 0;
  }
  else if (what == "emap") {
    HcalElectronicsMap* object = 0;
  }
  else if (what == "qie") {
    HcalQIEData* object = 0;
  }
  else if (what == "calibqie") {
    HcalCalibrationQIEData* object = 0;
  }
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
