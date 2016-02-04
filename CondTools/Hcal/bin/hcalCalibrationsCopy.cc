#include <stdlib.h>

#include <iostream>
#include <fstream>
#include <sstream>
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
#include "CondTools/Hcal/interface/HcalDbXml.h"
#include "CondTools/Hcal/interface/HcalDbOnline.h"
#include "CondTools/Hcal/interface/HcalDbTool.h"
//#include "CondTools/Hcal/interface/HcalDbPoolOCCI.h"
#include "CondFormats/HcalObjects/interface/HcalPedestals.h"
#include "CondFormats/HcalObjects/interface/HcalPedestalWidths.h"
#include "CondFormats/HcalObjects/interface/HcalGains.h"
#include "CondFormats/HcalObjects/interface/HcalGainWidths.h"
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/HcalObjects/interface/HcalQIEData.h"
#include "CondFormats/HcalObjects/interface/HcalCalibrationQIEData.h"

// CMSSW Message service
#include "FWCore/MessageService/interface/MessageServicePresence.h"

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

template <class T>
std::vector<HcalDetId> undefinedCells (const T& fData) {
  static std::vector<HcalDetId> result;
  if (result.size () <= 0) {
    HcalTopology topology;
    for (int eta = -63; eta < 64; eta++) {
      for (int phi = 0; phi < 128; phi++) {
	for (int depth = 1; depth < 5; depth++) {
	  for (int det = 1; det < 5; det++) {
	    HcalDetId cell ((HcalSubdetector) det, eta, phi, depth);
	    try {
	      fData.getValues (cell);
	    }
	    catch (...) {
	      if (topology.valid(cell)) result.push_back (cell);
	    }
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
    HcalPedestal item = HcalDbHardcode::makePedestal (*cell, false); // do not smear
    fPedestals->addValue (*cell, item.getValues ());
  }
  fPedestals->sort ();
}

void fillDefaults (HcalPedestalWidths*& fPedestals) {
  if (!fPedestals) {
    fPedestals = new HcalPedestalWidths;
    fPedestals->sort ();
  }
  std::vector<HcalDetId> cells = undefinedCells (*fPedestals);
  for (std::vector <HcalDetId>::const_iterator cell = cells.begin (); cell != cells.end (); cell++) {
    HcalPedestalWidth item = HcalDbHardcode::makePedestalWidth (*cell);
    fPedestals->setWidth (item);
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
    HcalGain item = HcalDbHardcode::makeGain (*cell, false); // do not smear
    fGains->addValue (*cell, item.getValues ());
  }
  fGains->sort ();
}

void fillDefaults (HcalGainWidths*& fGains) {
  if (!fGains) {
    fGains = new HcalGainWidths;
    fGains->sort ();
  }
  std::vector<HcalDetId> cells = undefinedCells (*fGains);
  for (std::vector <HcalDetId>::const_iterator cell = cells.begin (); cell != cells.end (); cell++) {
    HcalGainWidth item = HcalDbHardcode::makeGainWidth (*cell);
    fGains->addValue (*cell, item.getValues ());
  }
  fGains->sort ();
}

void fillDefaults (HcalElectronicsMap*& fMap) {
  if (!fMap) {
    fMap = new HcalElectronicsMap;
    fMap->sort ();
  }
  std::cerr << "Warning: fillDefaults (HcalElectronicsMap* fMap) is not implemented. Ignore." << std::endl;
}

void fillDefaults (HcalQIEData*& fObject) {
  if (!fObject) {
    fObject = new HcalQIEData;
    fObject->sort ();
  }
  HcalTopology topology;
  for (int eta = -63; eta < 64; eta++) {
    for (int phi = 0; phi < 128; phi++) {
      for (int depth = 1; depth < 5; depth++) {
	for (int det = 1; det < 5; det++) {
	  HcalDetId cell ((HcalSubdetector) det, eta, phi, depth);
	  if (topology.valid(cell)) {
	    HcalQIECoder item = HcalDbHardcode::makeQIECoder (cell); 
	    fObject->addCoder (item);
	  }
	}
      }
    }
  }
  fObject->sort ();
}

void fillDefaults (HcalCalibrationQIEData*& fObject) {
  if (!fObject) {
    fObject = new HcalCalibrationQIEData;
    fObject->sort ();
  }
  HcalTopology topology;
  for (int eta = -63; eta < 64; eta++) {
    for (int phi = 0; phi < 128; phi++) {
      for (int depth = 1; depth < 5; depth++) {
	for (int det = 1; det < 5; det++) {
	  HcalDetId cell ((HcalSubdetector) det, eta, phi, depth);
	  if (topology.valid(cell)) {
	    HcalCalibrationQIECoder item = HcalDbHardcode::makeCalibrationQIECoder (cell); 
	    fObject->addCoder (item);
	  }
	}
      }
    }
  }
  fObject->sort ();
}

void printHelp (const Args& args) {
  char buffer [1024];
  std::cout << "Tool to manipulate by Hcal Calibrations" << std::endl;
  std::cout << "    feedback -> ratnikov@fnal.gov" << std::endl;
  std::cout << "Use:" << std::endl;
  sprintf (buffer, " %s <what> <options> <parameters>\n", args.command ().c_str());
  std::cout << buffer;
  std::cout << "  where <what> is: \n    pedestals\n    gains\n    pwidths\n    gwidths\n    emap\n    qie\n    calibqie" << std::endl;
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

bool occiFile (const std::string fParam) {
  return fParam.find ("cms_val_lb.cern.ch") != std::string::npos &&
    fParam.find (':') == std::string::npos;
}

bool onlineFile (const std::string fParam) {
  return fParam.find ('@') != std::string::npos &&
    fParam.find ("cms_val_lb") == std::string::npos;
}

template <class T> bool copyObject (T* fObject, 
				    const std::string& fInput, const std::string& fInputTag, HcalDbTool::IOVRun fInputRun,
				    const std::string& fOutput, const std::string& fOutputTag, HcalDbTool::IOVRun fOutputRun,
				    bool fAppend,
				    unsigned long long fIovgmtbegin, unsigned long long fIovgmtend,
				    unsigned fNread, unsigned fNwrite, unsigned fNtrace,
				    bool fVerbose,
				    const char* fInputCatalog, const char* fOutputCatalog, bool fXmlAuth
				    ) {
  typedef std::vector <std::pair<HcalDbTool::IOVRun, T*> > Objects;

  bool result = false;
  time_t t0 = time (0);
  time_t t1 = t0;
  unsigned traceCounter = 0;
  HcalDbTool* poolDb = 0;
  HcalDbOnline* onlineDb = 0;
  //  HcalDbToolOCCI* occiDb = 0;
  Objects allInstances;
  while (traceCounter < fNread) {
    delete fObject;
    // get input
    if (defaultsFile (fInput)) {
      if (!traceCounter) std::cout << "USE INPUT: defaults" << std::endl;
      fillDefaults (fObject);
      result = true;
    }
    else if (asciiFile (fInput)) {
      if (!traceCounter) std::cout << "USE INPUT: ASCII: " << fInput << std::endl;
      std::ifstream inStream (fInput.c_str ());
      fObject = new T;
      HcalDbASCIIIO::getObject (inStream, fObject); 
      result = true;
    }
    else if (dbFile (fInput)) {
      if (!traceCounter) std::cout << "USE INPUT: Pool: " << fInput << "/" << fInputRun << std::endl;
      if (!poolDb) poolDb = new HcalDbTool (fInput, fVerbose, fXmlAuth, fInputCatalog);
      if (fInputRun > 0) {
	fObject = new T;
	result = poolDb->getObject (fObject, fInputTag, fInputRun);
      }
      else { // copy all instances
	std::cout << "Copy all instances... " << std::endl;
	cond::IOV iov;
	if (poolDb->getObject (&iov, fInputTag)) {
	  IOVCollection::const_iterator iovi = iov.iov.begin ();
	  for (; iovi != iov.iov.end (); iovi++) {
	    IOVRun iovMax = iovi->first;
	    if (fVerbose) {
	      std::cout << "fetching object for run " << iovMax << std::endl;
	    }
	    T* object = new T;
	    if (!poolDb->getObject (object, fInputTag, iovMax)) {
	      std::cerr << "Failed to fetch object..." << std::endl;
	      result = false;
	      delete object;
	      break;
	    }
	    allInstances.push_back (std::make_pair (iovMax, object));
	    std::cerr << '.';
	  }
	  if (iovi == iov.iov.end ()) result = true;
	}
	else {
	  std::cerr << "can not find IOV for tag " << fInputTag << std::endl;
	  result = false;
	}
      }
    }
    else if (onlineFile (fInput)) {
      if (!traceCounter) std::cout << "USE INPUT: Online: " << fInput << std::endl;
      if (!onlineDb) onlineDb = new HcalDbOnline (fInput, fVerbose);
      if (fInputRun > 0) {
	fObject = new T;
	result = onlineDb->getObject (fObject, fInputTag, fInputRun);
      }
      else { // copy all instances
	std::cout << "Copy all instances... " << std::endl;
	std::vector<HcalDbOnline::IntervalOV> iovs = onlineDb->getIOVs (fInputTag);
	for (unsigned i = 0; i < iovs.size ();i++) {
	  IOVRun iovMin = iovs[i].first;
	  if (fVerbose) {
	    std::cout << "fetching object for run " << iovMin << std::endl;
	  }
	  T* object = new T;
	  if (!onlineDb->getObject (object, fInputTag, iovMin)) {
	    std::cerr << "Failed to fetch object..." << std::endl;
	    result = false;
	    delete object;
	    break;
	  }
	  allInstances.push_back (std::make_pair (iovMin, object));
	  std::cerr << '.';
	}
	result = true;
      }
    }
 //    else if (occiFile (fInput)) {
//       if (!traceCounter) std::cout << "USE INPUT: OCCI" << std::endl;
//       if (!occiDb) occiDb = new HcalDbPoolOCCI (fInput);
//       fObject = new T;
//       result = occiDb->getObject (fObject, fInputTag, fInputRun);
//     }
    traceCounter++;
    fInputRun++;
    if (fNtrace && !(traceCounter % fNtrace)) {
      time_t t = time (0);
      std::cout << "read transaction: " << traceCounter << " time: " << t - t0 << " dtime: " << t - t1 << std::endl;
      t1 = t;
    }
  }
  delete poolDb;
  delete onlineDb;
  poolDb = 0;
  onlineDb = 0;
  if (result) {
    t0 = time (0);
    t1 = t0;
    traceCounter = 0;
    T* object = 0;
    while (traceCounter < fNwrite) {
      delete object;
      object = fObject ? new T (*fObject) : 0; // copy original object
      if (asciiFile (fOutput)) {
	if (!traceCounter) std::cout << "USE OUTPUT: ASCII: " << fOutput << std::endl;
	if (fObject && allInstances.empty ()) {
	  std::ofstream outStream (fOutput.c_str ());
	  HcalDbASCIIIO::dumpObject (outStream, *object);
	}
	else {
 	  for (unsigned i = 0; i < allInstances.size (); i++) {
	    if (fVerbose) {
 	      std::cout << "Storing object for run " << allInstances[i].first << std::endl;
	    }
	    std::ostringstream outName;
	    unsigned ipos = fOutput.find (".txt");
	    if (ipos == std::string::npos) {
	      outName << fOutput << "_" << allInstances[i].first;
	    }
	    else {
	      outName << std::string (fOutput, 0, ipos) << "_" << allInstances[i].first << ".txt";
	    }
	    std::ofstream outStream (outName.str().c_str ());
	    object = allInstances[i].second;
 	    HcalDbASCIIIO::dumpObject (outStream, *object);
	    delete object;
 	    allInstances[i].second = 0;
	    std::cerr << '.';
 	  }
	}
      }
      else if (xmlFile (fOutput)) {
	if (!traceCounter) std::cout << "USE OUTPUT: XML: " << fOutput << std::endl;
	std::ofstream outStream (fOutput.c_str ());
	HcalDbXml::dumpObject (outStream, fOutputRun, fIovgmtbegin, fIovgmtend, fOutputTag, *object);
	outStream.close ();
	std::cout << "close file\n";
      }
      else if (dbFile (fOutput)) { //POOL
	if (!traceCounter) std::cout << "USE OUTPUT: Pool: " << fOutput << '/' << fOutputRun << std::endl;
	if (!poolDb) poolDb = new HcalDbTool (fOutput, fVerbose, fXmlAuth, fOutputCatalog);
	if (fOutputRun > 0) {
	  poolDb->putObject (object, fOutputTag, fOutputRun, fAppend);
	  object = 0; // owned by POOL
	}
	else {
 	  for (unsigned i = 0; i < allInstances.size (); i++) {
	    if (fVerbose) {
 	      std::cout << "Storing object for run " << allInstances[i].first << std::endl;
	    }
 	    poolDb->putObject (allInstances[i].second, fOutputTag, allInstances[i].first, fAppend);
 	    allInstances[i].second = 0;
	    std::cerr << '.';
 	  }
	}
      }
      traceCounter++;
      fOutputRun++;
      if (fNtrace && !(traceCounter % fNtrace)) {
	time_t t = time (0);
	std::cout << "write transaction: " << traceCounter << " time: " << t - t0 << " dtime: " << t - t1 << std::endl;
	t1 = t;
      }
    }
    delete poolDb;
    poolDb = 0;
  }
  return result;
}

int main (int argn, char* argv []) {
  // start message service
  edm::service::MessageServicePresence my_message_service;

  Args args;
  args.defineParameter ("-input", "DB connection string, POOL format, or .txt file, or defaults");
  args.defineParameter ("-output", "DB connection string, POOL format, or .txt, or .xml file");
  args.defineParameter ("-inputrun", "run # for which constands should be made");
  args.defineParameter ("-inputtag", "tag for the input constants set");
  args.defineParameter ("-inputcatalog", "catalog for POOL DB <$POOL_CATALOG>");
  args.defineParameter ("-outputrun", "run # for which constands should be dumped");
  args.defineParameter ("-outputtag", "tag for the output constants set");
  args.defineParameter ("-outputcatalog", "catalog for POOL DB <$POOL_CATALOG>");
  args.defineParameter ("-iovgmtbegin", "start time for online IOV <outputrun>");
  args.defineParameter ("-iovgmtend", "end time for online IOV <0>");
  args.defineParameter ("-nread", "repeat input that many times with increasing run# <1>");
  args.defineParameter ("-nwrite", "repeat output that many times with increasing run# <1>");
  args.defineParameter ("-trace", "trace time every that many operations <false>");
  args.defineOption ("-help", "this help");
  args.defineOption ("-online", "interpret input DB as an online DB");
  args.defineOption ("-xmlauth", "use XML authorization <false>");
  args.defineOption ("-append", "Strip previous IOV, make this IOV open (POOL DB) <false>");
  args.defineOption ("-verbose", "makes program verbose <false>");
  
  args.parse (argn, argv);
  
  std::vector<std::string> arguments = args.arguments ();

  if (arguments.size () < 1 || args.optionIsSet ("-help")) {
    printHelp (args);
    return -1;
  }

  std::string input = args.getParameter ("-input");
  std::string output = args.getParameter ("-output");
  
  HcalDbTool::IOVRun inputRun = args.getParameter ("-inputrun").empty () ? 0 : strtoull (args.getParameter ("-inputrun").c_str (), 0, 0);
  HcalDbTool::IOVRun outputRun = args.getParameter ("-outputrun").empty () ? 0 : strtoll (args.getParameter ("-outputrun").c_str (), 0, 0);
  std::string inputTag = args.getParameter ("-inputtag");
  std::string outputTag = args.getParameter ("-outputtag");

  unsigned long long iovgmtbegin = args.getParameter ("-iovgmtbegin").empty () ? outputRun : strtoull (args.getParameter ("-iovgmtbegin").c_str (), 0, 0);
  unsigned long long iovgmtend = args.getParameter ("-iovgmtend").empty () ? 0 : strtoull (args.getParameter ("-iovgmtend").c_str (), 0, 0);

  unsigned nread = args.getParameter ("-nread").empty () ? 1 : atoi (args.getParameter ("-nread").c_str ());
  unsigned nwrite = args.getParameter ("-nwrite").empty () ? 1 : atoi (args.getParameter ("-nwrite").c_str ());
  unsigned trace = args.getParameter ("-trace").empty () ? 0 : atoi (args.getParameter ("-trace").c_str ());

  const char* inputCatalog = args.getParameter ("-inputcatalog").empty () ? 0 : args.getParameter ("-inputcatalog").c_str();
  const char* outputCatalog = args.getParameter ("-outputcatalog").empty () ? 0 : args.getParameter ("-outputcatalog").c_str();

  bool xmlAuth = args.optionIsSet ("-xmlauth");
  bool append = args.optionIsSet ("-append");

  bool verbose = args.optionIsSet ("-verbose");


  std::string what = arguments [0];

  if (what == "pedestals") {
    HcalPedestals* object = 0;
    copyObject (object, input, inputTag, inputRun, output, outputTag, outputRun, append, iovgmtbegin, iovgmtend, nread, nwrite, trace, verbose, inputCatalog, outputCatalog, xmlAuth);
  }
  else if (what == "gains") {
    HcalGains* object = 0;
    copyObject (object, input, inputTag, inputRun, output, outputTag, outputRun, append, iovgmtbegin, iovgmtend, nread, nwrite, trace, verbose, inputCatalog, outputCatalog, xmlAuth);
  }
  else if (what == "pwidths") {
    HcalPedestalWidths* object = 0;
    copyObject (object, input, inputTag, inputRun, output, outputTag, outputRun, append, iovgmtbegin, iovgmtend, nread, nwrite, trace, verbose, inputCatalog, outputCatalog, xmlAuth);
  }
  else if (what == "gwidths") {
    HcalGainWidths* object = 0;
    copyObject (object, input, inputTag, inputRun, output, outputTag, outputRun, append, iovgmtbegin, iovgmtend, nread, nwrite, trace, verbose, inputCatalog, outputCatalog, xmlAuth);
  }
  else if (what == "emap") {
    HcalElectronicsMap* object = 0;
    copyObject (object, input, inputTag, inputRun, output, outputTag, outputRun, append, iovgmtbegin, iovgmtend, nread, nwrite, trace, verbose, inputCatalog, outputCatalog, xmlAuth);
  }
  else if (what == "qie") {
    HcalQIEData* object = 0;
    copyObject (object, input, inputTag, inputRun, output, outputTag, outputRun, append, iovgmtbegin, iovgmtend, nread, nwrite, trace, verbose, inputCatalog, outputCatalog, xmlAuth);
  }
  else if (what == "calibqie") {
    HcalCalibrationQIEData* object = 0;
    copyObject (object, input, inputTag, inputRun, output, outputTag, outputRun, append, iovgmtbegin, iovgmtend, nread, nwrite, trace, verbose, inputCatalog, outputCatalog, xmlAuth);
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
