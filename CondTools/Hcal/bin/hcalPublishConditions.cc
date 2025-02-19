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
  std::cout << "Tool to convert RAW HCAL conditions into standard offline accessible ones" << std::endl;
  std::cout << "    feedback -> ratnikov@fnal.gov" << std::endl;
  std::cout << "Use:" << std::endl;
  sprintf (buffer, " %s <what> <options> <parameters>\n", args.command ().c_str());
  std::cout << buffer;
  std::cout << "  where <what> is: \n    pedestals\n    gains\n    pwidths\n    gwidths\n    emap\n    qie\n    calibqie" << std::endl;
  args.printOptionsHelp ();
}


bool publishObjects (const std::string& fInputDb, const std::string& fInputTag, 
		 const std::string& fOutputDb, const std::string& fOutputTag,
		 bool fVerbose) {
  HcalDbTool db (fInputDb, fVerbose);
  cond::IOV inputIov;
  cond::IOV outputIov;
  std::vector<std::string> allTags = db.metadataAllTags ();
  for (unsigned i = 0; i < allTags.size(); i++) {
    if (allTags[i] == fInputTag) {
      if (!db.getObject (&inputIov, fInputTag)) {
	std::cerr << "copyObject-> Can not get IOV for input tags " << fInputTag << std::endl;
	return false;
      }
    }
    if (allTags[i] == fOutputTag) {
      if (!db.getObject (&outputIov, fOutputTag)) {
	std::cerr << "copyObject-> Can not get IOV for output tags " << fOutputTag << std::endl;
	return false;
      }
    }
  }
  if (fVerbose) {
    std::vector <IntervalOV> allIOVs = allIOV (inputIov);
    std::cout << " all IOVs available for input tag " << fInputTag << " in CMS POOL DB instance: " << fInputDb << std::endl;
    for (unsigned i = 0; i < allIOVs.size(); i++) {
      std::cout << "[ " << allIOVs[i].first << " .. " << allIOVs[i].second << " )  " << inputIov.iov [allIOVs[i].second] << std::endl;
    }
    allIOVs = allIOV (outputIov);
    std::cout << "\n all IOVs available for output tag " << fOutputTag << " in CMS POOL DB instance: " << fInputDb << std::endl;
    for (unsigned i = 0; i < allIOVs.size(); i++) {
      std::cout << "[ " << allIOVs[i].first << " .. " << allIOVs[i].second << " )  " << outputIov.iov [allIOVs[i].second] << std::endl;
    }
  }
  
  // first check that all objects of output are contained in input
  IOVCollection::const_iterator iovIn = inputIov.iov.begin ();
  if (iovIn == inputIov.iov.end ()) {
    std::cerr << "Input IOV is empty - nothing to do" << std::endl;
    return true;
  }
  std::string inputToken = iovIn->second;
  iovIn++;
  IOVCollection::const_iterator iovOut = outputIov.iov.begin ();
  for (; ; iovOut++, iovIn++) {
    if (iovIn == inputIov.iov.end ()) {
      if (++iovOut == outputIov.iov.end ()) {
	std::cerr << "Nothing to update" << std::endl;
	return true;
      }
      else {
	std::cerr << "List of input IOV is too short" << std::endl;
	return false;
      }
    }
    if (iovOut == outputIov.iov.end ()) { // empty output
      outputIov.iov [iovIn->first] = inputToken;
      inputToken = iovIn->second;
      break;
    }
    if (inputToken != iovOut->second) {
      std::cerr << "Mismatched tokens: \n  in: " << iovIn->second << "\n out: " << iovOut->second << std::endl;
      return false;
    }
    
    // is it the open end?
    IOVCollection::const_iterator iovOut2 = iovOut;
    if (++iovOut2 == outputIov.iov.end ()) {
      outputIov.iov.erase (iovOut->first);
      outputIov.iov [iovIn->first] = inputToken;
      inputToken = iovIn->second;
      break;
    }
    if (iovIn->first != iovOut->first) {
      std::cerr << "Mismatched runs: in: " << iovIn->first << ", out: " << iovOut->first << std::endl;
      return false;
    }
    
    inputToken = iovIn->second;
  }
  std::cout << "Good! Input and output does match" << std::endl;
  
  for (iovIn++; iovIn != inputIov.iov.end (); iovIn++) {
    IOVRun run = iovIn->first;
    outputIov.iov [run] = inputToken;
    inputToken = iovIn->second;
  }
  // last open token
  outputIov.iov [edm::Timestamp::endOfTime().value()] = inputToken;
  
  if (fVerbose) {
    std::vector <IntervalOV> allIOVs = allIOV (outputIov);
    std::cout << "\n Done! All IOVs available for output tag " << fOutputTag << " in CMS POOL DB instance: " << fInputDb << std::endl;
    for (unsigned i = 0; i < allIOVs.size(); i++) {
      std::cout << "[ " << allIOVs[i].first << " ... " << allIOVs[i].second << " )  " << outputIov.iov [allIOVs[i].second] << std::endl;
    }
  }
  return db.putObject (&outputIov, fOutputTag);
}


int main (int argn, char* argv []) {

  Args args;
  args.defineParameter ("-input", "DB connection string, POOL format, i.e. oracle://devdb10/CMS_COND_HCAL");
  args.defineParameter ("-inputtag", "tag for raw conditions");
  args.defineParameter ("-output", "DB connection string, POOL format, i.e. oracle://devdb10/CMS_COND_HCAL");
  args.defineParameter ("-outputtag", "tag for production conditions");
  args.defineOption ("-help", "this help");
  args.defineOption ("-verbose", "verbose");
  
  args.parse (argn, argv);
  
  std::vector<std::string> arguments = args.arguments ();

  if (arguments.size () < 1 || args.optionIsSet ("-help")) {
    printHelp (args);
    return -1;
  }

  std::string inputdb = args.getParameter ("-input");
  std::string inputtag = args.getParameter ("-inputtag");
  std::string outputdb = args.getParameter ("-output");
  std::string outputtag = args.getParameter ("-outputtag");
  bool verbose = args.optionIsSet ("-verbose");

  publishObjects (inputdb, inputtag, outputdb, outputtag, verbose);
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
