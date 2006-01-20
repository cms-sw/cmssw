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


// Hcal calibrations
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbHardcode.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbXml.h"
#include "CondTools/Hcal/interface/HcalDbOnline.h"
#include "CondTools/Hcal/interface/HcalDbPool.h"
#include "CondFormats/HcalObjects/interface/HcalPedestals.h"
#include "CondFormats/HcalObjects/interface/HcalPedestalWidths.h"
#include "CondFormats/HcalObjects/interface/HcalGains.h"
#include "CondFormats/HcalObjects/interface/HcalGainWidths.h"
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"
#include "CondFormats/HcalObjects/interface/HcalQIEShape.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/HcalObjects/interface/HcalQIEData.h"

//using namespace cms;

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

template <class T>
std::vector<HcalDetId> undefinedCells (const T& fData) {
  static std::vector<HcalDetId> result;
  if (result.size () <= 0) {
    HcalTopology topology;
    for (int eta = -50; eta < 50; eta++) {
      for (int phi = 0; phi < 100; phi++) {
	for (int depth = 1; depth < 5; depth++) {
	  for (int det = 1; det < 5; det++) {
	    HcalDetId cell ((HcalSubdetector) det, eta, phi, depth);
	    if (topology.valid(cell) && !fData.getValues (cell.rawId())) result.push_back (cell);
	  }
	}
      }
    }
  }
  return result;
}

void fillDefaults (HcalPedestals*& fPedestals) {
  if (!fPedestals) {
    fPedestals = new HcalPedestals;
    fPedestals->sort ();
  }
  std::vector<HcalDetId> cells = undefinedCells (*fPedestals);
  for (std::vector <HcalDetId>::const_iterator cell = cells.begin (); cell != cells.end (); cell++) {
    HcalPedestal item = HcalDbHardcode::makePedestal (*cell);
    fPedestals->addValue (*cell, item.getValues ());
  }
  fPedestals->sort ();
}

void fillDefaults (HcalGains*& fGains) {
  if (!fGains) {
    fGains = new HcalGains;
    fGains->sort ();
  }
  std::vector<HcalDetId> cells = undefinedCells (*fGains);
  for (std::vector <HcalDetId>::const_iterator cell = cells.begin (); cell != cells.end (); cell++) {
    HcalGain item = HcalDbHardcode::makeGain (*cell);
    fGains->addValue (*cell, item.getValues ());
  }
  fGains->sort ();
}

void fillDefaults (HcalElectronicsMap* fMap) {
  std::cerr << "ERROR: fillDefaults (HcalElectronicsMap* fMap) is not implemented. Ignore." << std::endl;
}

void printHelp (const Args& args) {
  char buffer [1024];
  std::cout << "Tool to manipulate by Hcal Calibrations" << std::endl;
  std::cout << "    feedback -> ratnikov@fnal.gov" << std::endl;
  std::cout << "Use:" << std::endl;
  sprintf (buffer, " %s dump <what> <options> <parameters>\n", args.command ().c_str());
  std::cout << buffer;
  sprintf (buffer, " %s fill <what> <options> <parameters>\n", args.command ().c_str());
  std::cout << buffer;
  sprintf (buffer, " %s add <what> <options> <parameters>\n", args.command ().c_str());
  std::cout << buffer;
  std::cout << "  where <what> is: \n    pedestals\n    gains\n    emap\n" << std::endl;
  args.printOptionsHelp ();
}

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

bool onlineFile (const std::string fParam) {
  return fParam.find ('@') != std::string::npos;
}

int main (int argn, char* argv []) {

  Args args;
  args.defineParameter ("-input", "DB connection string, POOL format, or .txt file, or defaults");
  args.defineParameter ("-output", "DB connection string, POOL format, or .txt, or .xml file");
  args.defineParameter ("-inputrun", "run # for which constands should be made");
  args.defineParameter ("-inputtag", "tag for the input constants set");
  args.defineParameter ("-outputrun", "run # for which constands should be dumped");
  args.defineParameter ("-outputtag", "tag for the output constants set");
  args.defineOption ("-help", "this help");
  args.defineOption ("-online", "Interpret input DB as an online DB");
  
  args.parse (argn, argv);
  
  std::vector<std::string> arguments = args.arguments ();

  if (arguments.size () < 1 || args.optionIsSet ("-help")) {
    printHelp (args);
    return -1;
  }

  std::string input = args.getParameter ("-input");
  std::string output = args.getParameter ("-output");
  
  unsigned inputRun = args.getParameter ("-inputrun").empty () ? 0 : atoi (args.getParameter ("-inputrun").c_str ());
  unsigned outputRun = args.getParameter ("-outputrun").empty () ? 0 : atoi (args.getParameter ("-outputrun").c_str ());
  std::string inputTag = args.getParameter ("-inputtag");
  std::string outputTag = args.getParameter ("-outputtag");
  bool online = args.optionIsSet ("-online");

  bool getPedestals = arguments [0] == "pedestals";
  bool getGains = arguments [0] == "gains";
  bool emap = arguments [0] == "emap";

  std::string what = arguments [0];

  if (arguments [0] == "pedestals") {
    HcalPedestals* object = 0;
    bool readOK = false;
    // get input
    if (defaultsFile (input)) {
      std::cout << "USE INPUT: defaults" << std::endl;
      fillDefaults (object);
      readOK = true;
    }
    else if (asciiFile (input)) {
      std::cout << "USE INPUT: ASCII" << std::endl;
      std::ifstream inStream (input.c_str ());
      object = new HcalPedestals;
      HcalDbASCIIIO::getObject (inStream, object); 
      readOK = true;
    }
    else if (dbFile (input)) {
      std::cout << "USE INPUT: Pool" << std::endl;
      HcalDbPool poolDb (input);
      readOK = poolDb.getObject (object, inputTag, inputRun);
    }
    else if (onlineFile (input)) {
      std::cout << "USE INPUT: Online" << std::endl;
      HcalDbOnline onlineDb (input);
      std::auto_ptr<HcalPedestals> autoptr = onlineDb.getPedestals (inputTag);
      object = autoptr.release ();
      readOK = true;
    }
    if (readOK) {
      if (asciiFile (output)) {
	std::cout << "USE OUTPUT: ASCII" << std::endl;
	std::ofstream outStream (output.c_str ());
	HcalDbASCIIIO::dumpObject (outStream, *object); 
      }
      else if (xmlFile (output)) {
	std::cout << "USE OUTPUT: XML" << std::endl;
	std::ofstream outStream (output.c_str ());
	HcalDbXml::dumpObject (outStream, outputRun, outputTag, *object);
      }
      else if (dbFile (output)) { //POOL
	std::cout << "USE OUTPUT: Pool" << std::endl;
	HcalDbPool poolDb (output);
	poolDb.putObject (object, outputTag, outputRun);
      }
    }
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
