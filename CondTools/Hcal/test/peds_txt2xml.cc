#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>

#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"
#include "CondTools/Hcal/interface/HcalDbXml.h"
#include "CondTools/Hcal/interface/HcalDbTool.h"
#include "CondFormats/HcalObjects/interface/HcalPedestals.h"
#include "CondFormats/HcalObjects/interface/HcalPedestalWidths.h"
#include "CondFormats/HcalObjects/interface/HcalGains.h"
#include "CondFormats/HcalObjects/interface/HcalGainWidths.h"
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/HcalObjects/interface/HcalQIEData.h"
#include "CondFormats/HcalObjects/interface/HcalCalibrationQIEData.h"


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
  std::cout << "Tool to combine pedestals and widths into single xml file" << std::endl;
  std::cout << "    feedback -> ratnikov@fnal.gov" << std::endl;
  std::cout << "Use:" << std::endl;
  sprintf (buffer, " %s <options> <parameters>\n", args.command ().c_str());
  std::cout << buffer;
  args.printOptionsHelp ();
}

int main (int argn, char* argv []) {

  Args args;
  args.defineParameter ("-values", "ascii pedestals input");
  args.defineParameter ("-widths", "ascii widths input");
  args.defineParameter ("-output", "xml output");
  args.defineOption ("-help", "this help");
  args.defineParameter ("-outputrun", "run # for which constands should be dumped");
  args.defineParameter ("-outputtag", "tag for the output constants set");
  args.defineParameter ("-iovgmtbegin", "start time for online IOV <outputrun>");
  args.defineParameter ("-iovgmtend", "end time for online IOV <0>");

  
  args.parse (argn, argv);
  
  std::vector<std::string> arguments = args.arguments ();

  if (args.optionIsSet ("-help")) {
    printHelp (args);
    return -1;
  }

  std::string peds = args.getParameter ("-values");
  std::string widths = args.getParameter ("-widths");
  std::string output = args.getParameter ("-output");
  HcalDbTool::IOVRun outputRun = args.getParameter ("-outputrun").empty () ? 0 : strtoll (args.getParameter ("-outputrun").c_str (), 0, 0);
  std::string outputTag = args.getParameter ("-outputtag");
  unsigned long long iovgmtbegin = args.getParameter ("-iovgmtbegin").empty () ? outputRun : strtoull (args.getParameter ("-iovgmtbegin").c_str (), 0, 0);
  unsigned long long iovgmtend = args.getParameter ("-iovgmtend").empty () ? 0 : strtoull (args.getParameter ("-iovgmtend").c_str (), 0, 0);


  std::ifstream inStream (peds.c_str ());
  HcalPedestals object;
  HcalDbASCIIIO::getObject (inStream, &object);
  object.sort ();
  std::ifstream inStream2 (widths.c_str ());
  HcalPedestalWidths objectW;
  HcalDbASCIIIO::getObject (inStream2, &objectW);
  objectW.sort ();
  std::ofstream outStream (output.c_str ());
  HcalDbXml::dumpObject (outStream, outputRun, iovgmtbegin, iovgmtend, outputTag, object, objectW);

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


