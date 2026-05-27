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
#include "CondFormats/HcalObjects/interface/HcalFloats.h"

using namespace cmdline;

static void print_usage(const char *progname) {
  std::cout << "\nUsage: " << progname << " [-a] [-t] [-p precision] inputfile outputfile\n\n"
            << "Switch -a can be used to specify that the input file is a boost\n"
            << "archive. If this switch is not provided, the input file is assumed\n"
            << "to be in a simple human-readable text format. For the description of\n"
            << "this format and more detals on pulse shape representation, see file\n"
            << "\"readme_HcalInterpolatedPulseMap.txt\".\n\n"
            << "Switch -t can be used to specify that the output file should be written\n"
            << "in the same simple text format (otherwise it will be a boost archive).\n"
            << "If option -t is provided, option -p can be used to specify the precision\n"
            << "with which floating point numbers will be written out (default value of\n"
            << "this option is " << HcalFloats::kDefaultPrecision
            << "). Without -t, option -p will be ignored (boost archives\n"
            << "are always written in full precision).\n"
            << std::endl;
}

static bool is_text_file(const std::string &filename) {
  bool usingTxt = false;
  const unsigned outlen = strlen(filename.c_str());
  if (outlen >= 4)
    usingTxt = strcmp(".txt", filename.c_str() + outlen - 4) == 0;
  return usingTxt;
}

static bool read_pulses_from_archive(const std::string &filename, HcalInterpolatedPulseMap &pulseMap) {
  std::ios_base::openmode mode = std::ios::in;
  const bool usingTxt = is_text_file(filename);
  if (!usingTxt) {
    mode |= std::ios::binary;
  }
  std::ifstream is(filename, mode);
  if (!is.is_open()) {
    std::cerr << "Failed to open input file " << filename << std::endl;
    return false;
  }
  if (usingTxt) {
    boost::archive::text_iarchive ar(is);
    ar & pulseMap;
  } else {
    eos::portable_iarchive ar(is);
    ar & pulseMap;
  }
  return true;
}

static bool write_pulses_to_archive(const std::string &filename, const HcalInterpolatedPulseMap &pulseMap) {
  std::ios_base::openmode mode = std::ios::out;
  const bool usingTxt = is_text_file(filename);
  if (!usingTxt) {
    mode |= std::ios::binary;
  }
  std::ofstream of(filename, mode);
  if (!of.is_open()) {
    std::cerr << "Failed to open output file " << filename << std::endl;
    return false;
  }
  if (usingTxt) {
    boost::archive::text_oarchive ar(of);
    ar & pulseMap;
  } else {
    eos::portable_oarchive ar(of);
    ar & pulseMap;
  }
  return true;
}

int main(int argc, char *argv[]) {
  CmdLine cmdline(argc, argv);

  if (argc == 1) {
    print_usage(cmdline.progname());
    return 0;
  }

  unsigned precision = HcalFloats::kDefaultPrecision;
  std::string inputfile, outputfile;
  bool inputIsArchive = false;
  bool dumpToText = false;

  try {
    inputIsArchive = cmdline.has("-a");
    dumpToText = cmdline.has("-t");
    cmdline.option("-p") >> precision;

    cmdline.optend();
    if (cmdline.argc() != 2)
      throw CmdLineError("wrong number of command line arguments");
    cmdline >> inputfile >> outputfile;
  } catch (const CmdLineError &e) {
    std::cerr << "Error in " << cmdline.progname() << ": " << e.str() << std::endl;
    return 1;
  }

  HcalInterpolatedPulseMap pulseMap;
  if (inputIsArchive)
    read_pulses_from_archive(inputfile, pulseMap);
  else
    pulseMap.readFromTxt(inputfile);

  // Are we performing a simple text dump?
  if (dumpToText) {
    pulseMap.dumpToTxt(outputfile, precision);
    return 0;
  }

  // Write the object out
  if (!write_pulses_to_archive(outputfile, pulseMap))
    return 1;

  // Read it back in
  HcalInterpolatedPulseMap map2;
  if (!read_pulses_from_archive(outputfile, map2))
    return 1;

  // Make sure that they are the same
  assert(pulseMap == map2);

  return 0;
}
