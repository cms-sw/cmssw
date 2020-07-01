#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>

#include <string>
#include <cstring>

#include "CondFormats/Serialization/interface/eos/portable_iarchive.hpp"
#include "CondFormats/Serialization/interface/eos/portable_oarchive.hpp"

#include "CondTools/Hcal/interface/make_HBHENegativeEFilter.h"

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

using namespace std;

static void print_usage(const char *progname) {
  cout << "\nUsage: " << progname << " which outputfile\n\n"
       << "Argument \"which\" must be 0 or 1:\n\n"
       << "  0 -- write Alexander Toropin's filter\n"
       << "  1 -- write a dummy (all pass) filter\n"
       << endl;
}

int main(int argc, char *argv[]) {
  // Parse input arguments
  if (argc != 3) {
    print_usage(argv[0]);
    return argc != 1;
  }
  const string which(argv[1]);
  const char *outputfile = argv[2];

  // Are we using a text file as output?
  bool usingTxt = false;
  std::ios_base::openmode mode = std::ios::binary;
  {
    const unsigned outlen = strlen(outputfile);
    if (outlen >= 4)
      usingTxt = strcmp(".txt", outputfile + outlen - 4) == 0;
    if (usingTxt)
      mode = std::ios_base::openmode();
  }

  // Make the filter
  std::unique_ptr<HBHENegativeEFilter> f1;
  if (which == "0")
    f1 = make_HBHENegativeEFilter();
  else if (which == "1")
    f1 = std::make_unique<HBHENegativeEFilter>();
  else {
    cerr << "Error: invalid filter code \"" << which << "\"." << endl;
    print_usage(argv[0]);
    return 1;
  }

  // Write it out
  {
    std::ofstream of(outputfile, mode);
    if (!of.is_open()) {
      cerr << "Failed to open file " << outputfile << endl;
      return 1;
    }
    if (usingTxt) {
      boost::archive::text_oarchive ar(of);
      ar &*f1;
    } else {
      eos::portable_oarchive ar(of);
      ar &*f1;
    }
  }

  // Read it back in
  HBHENegativeEFilter f2;
  {
    std::ifstream is(outputfile, mode);
    if (usingTxt) {
      boost::archive::text_iarchive ar(is);
      ar &f2;
    } else {
      eos::portable_iarchive ar(is);
      ar &f2;
    }
  }

  // Make sure that they are the same
  assert(*f1 == f2);

  return 0;
}
