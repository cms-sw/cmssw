#include <iostream>
#include <stdexcept>
#include <sstream>
#include "TString.h"
#include "PhysicsTools/FWLite/interface/CommandLineParser.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/HcalLutManager.h"
#include "FWCore/Utilities/interface/FileInPath.h"

using namespace std;

void mergeLUTs(const char *flist, const char *out) {
  LutXml xmls;
  stringstream ss(flist);
  while (ss.good()) {
    string file;
    ss >> file;
    xmls += LutXml(file);
  }
  xmls.write(out);
}

void dumpLutDiff(LutXml &xmls1, LutXml &xmls2, bool testFormat = true, int detail = 0) {
  const int ndet = 5;
  const char *DET[ndet] = {"HB", "HE", "HO", "HF", "HT"};
  const int dtype[ndet] = {0, 1, 2, 3, 4};

  const int nvar = 4;
  enum vtype { total, extra, zeros, match };

  std::array<int, nvar> n[ndet];

  for (auto &d : n) {
    for (auto &v : d) {
      v = 0;
    }
  }

  for (auto &x1 : xmls1) {
    HcalGenericDetId id(x1.first);
    auto x2 = xmls2.find(id.rawId());
    auto subdet = id.genericSubdet();
    if (subdet == 0 or subdet == 6)
      continue;  //'empty' or 'other'

    auto &m = n[subdet - 1];

    m[total]++;
    if (x2 == xmls2.end()) {
      m[extra]++;
      if (testFormat)
        cout << "Extra detId: " << id << endl;
      else
        continue;
    }

    const auto &lut1 = x1.second;
    size_t size = lut1.size();

    bool zero = true;
    for (auto &i : lut1) {
      if (i > 0) {
        zero = false;
        break;
      }
    }
    if (zero) {
      m[zeros]++;
      if (detail == 1 and testFormat) {
        cout << "Zero LUT: " << id << endl;
      }
    }

    if (testFormat)
      continue;

    const auto &lut2 = x2->second;
    bool good = size == lut2.size();
    for (size_t i = 0; i < size and good; ++i) {
      if (lut1[i] != lut2[i]) {
        good = false;
        if (detail == 2) {
          cout << Form("Mismatach in index=%3d, %4d!=%4d, ", int(i), lut1[i], lut2[i]) << id << endl;
        }
      }
    }
    if (good)
      m[match]++;
  }

  if (testFormat) {
    cout << Form("%3s:  %8s  %8s  %8s", "Det", "total", "zeroes", "extra") << endl;
    for (auto i : dtype)
      cout << Form("%3s:  %8d  %8d  %8d", DET[i], n[i][total], n[i][zeros], n[i][extra]) << endl;
    cout << "--------------------------------------------" << endl;
  } else {
    bool good = true;
    for (auto &d : n) {
      if (d[total] != d[match]) {
        good = false;
      }
    }
    cout << Form("%3s:  %8s  %8s  %8s", "Det", "total", "match", "mismatch") << endl;
    for (auto i : dtype)
      cout << Form("%3s:  %8d  %8d  %8d", DET[i], n[i][total], n[i][match], n[i][total] - n[i][match]) << endl;
    cout << "--------------------------------------------" << endl;
    cout << (good ? "PASS!" : "FAIL!") << endl;
  }
}

int main(int argc, char **argv) {
  optutl::CommandLineParser parser("runTestParameters");
  parser.parseArguments(argc, argv, true);
  if (argc < 2) {
    std::cerr << "runTest: missing input command" << std::endl;
  } else if (strcmp(argv[1], "merge") == 0) {
    std::string flist_ = parser.stringValue("storePrepend");
    std::string out_ = parser.stringValue("outputFile");
    mergeLUTs(flist_.c_str(), out_.c_str());
  } else if (strcmp(argv[1], "diff") == 0) {
    auto files = parser.stringVector("inputFiles");
    auto detail = parser.integerValue("section");

    LutXml xmls1(edm::FileInPath(files[0]).fullPath());
    LutXml xmls2(edm::FileInPath(files[1]).fullPath());

    xmls1.create_lut_map();
    xmls2.create_lut_map();

    cout << files[0] << endl;
    dumpLutDiff(xmls1, xmls2, true, detail);

    cout << files[1] << endl;
    dumpLutDiff(xmls2, xmls1, true, detail);

    cout << "Comparison" << endl;
    dumpLutDiff(xmls1, xmls2, false, detail);
  } else if (strcmp(argv[1], "create-lut-loader") == 0) {
    std::string _file_list = parser.stringValue("outputFile");
    std::string _tag = parser.stringValue("tag");
    std::string _comment = parser.stringValue("storePrepend");
    const std::string &_prefix = _tag;
    std::string _version = "1";
    int _subversion = 0;
    HcalLutManager manager;
    manager.create_lut_loader(_file_list, _prefix, _tag, _comment, _tag, _subversion);
  } else {
    throw std::invalid_argument(Form("Unknown command: %s", argv[1]));
  }

  return 0;
}
