#include <cassert>
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>

#include "CondTools/Hcal/interface/CmdLine.h"

#include "CondFormats/Serialization/interface/eos/portable_iarchive.hpp"
#include "CondFormats/Serialization/interface/eos/portable_oarchive.hpp"

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include "CondFormats/HcalObjects/interface/HcalInterpolatedPulseMap.h"

using namespace std;
using namespace cmdline;

static void print_usage(const char *progname) {
  cout << "\nUsage: " << progname << " [-a] [-p precision] inputfile outputfile\n\n"
       << "Option -p can be used to specify the floating point precision.\n"
       << "If this option is provided, the output file will be a text file\n"
       << "written out using the precision requested (value of 0 stands for\n"
       << "C++ default precision). If this option is not provided, the output\n"
       << "file will be a boost archive (text or binary).\n\n"
       << "Switch -a can be used to specify that the input file is a boost\n"
       << "archive. Should normally be combined with the -p option.\n"
       << endl;
}

static bool is_text_file(const std::string &filename) {
  bool usingTxt = false;
  const unsigned outlen = strlen(filename.c_str());
  if (outlen >= 4)
    usingTxt = strcmp(".txt", filename.c_str() + outlen - 4) == 0;
  return usingTxt;
}

int main(int argc, char *argv[]) {
  CmdLine cmdline(argc, argv);

  if (argc == 1) {
    print_usage(cmdline.progname());
    return 0;
  }

  int precision = -1;
  string inputfile, outputfile;
  bool inputIsArchive;

  try {
    inputIsArchive = cmdline.has("-a");
    cmdline.option("-p") >> precision;

    cmdline.optend();
    if (cmdline.argc() != 2)
      throw CmdLineError("wrong number of command line arguments");
    cmdline >> inputfile >> outputfile;
  } catch (const CmdLineError &e) {
    cerr << "Error in " << cmdline.progname() << ": " << e.str() << std::endl;
    return 1;
  }

  HcalInterpolatedPulseMap pulseMap;
  if (inputIsArchive) {
    std::ios_base::openmode mode = std::ios::in;
    const bool usingTxt = is_text_file(inputfile);
    if (!usingTxt)
      mode |= std::ios::binary;
    {
      std::ifstream is(inputfile, mode);
      if (usingTxt) {
        boost::archive::text_iarchive ar(is);
        ar & pulseMap;
      } else {
        eos::portable_iarchive ar(is);
        ar & pulseMap;
      }
    }
  } else
    pulseMap.readFromTxt(inputfile);

  // Are we performing a simple text dump?
  if (precision >= 0) {
    pulseMap.dumpToTxt(outputfile, precision);
    return 0;
  }

  // Are we using a text archive as output?
  const bool usingTxt = is_text_file(outputfile);

  // Write the object out
  {
    std::ios_base::openmode mode = std::ios::out;
    if (!usingTxt)
      mode |= std::ios::binary;
    std::ofstream of(outputfile, mode);
    if (!of.is_open()) {
      cerr << "Failed to open file " << outputfile << endl;
      return 1;
    }
    if (usingTxt) {
      boost::archive::text_oarchive ar(of);
      ar & pulseMap;
    } else {
      eos::portable_oarchive ar(of);
      ar & pulseMap;
    }
  }

  // Read it back in
  HcalInterpolatedPulseMap map2;
  {
    std::ios_base::openmode mode = std::ios::in;
    if (!usingTxt)
      mode |= std::ios::binary;
    std::ifstream is(outputfile, mode);
    if (usingTxt) {
      boost::archive::text_iarchive ar(is);
      ar & map2;
    } else {
      eos::portable_iarchive ar(is);
      ar & map2;
    }
  }

  // Make sure that they are the same
  assert(pulseMap == map2);

  return 0;
}
