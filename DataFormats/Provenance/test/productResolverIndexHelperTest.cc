#include "DataFormats/Provenance/interface/ProductResolverIndexHelper.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "FWCore/Utilities/interface/CPUTimer.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/ProductKindOfType.h"

#include "TClass.h"

#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <istream>
#include <vector>

// This program will run and time the ProductResolverIndexHelper class.
// Before running it one needs to create a text file named
// "log3" which contains a list of products. I generated
// this file as follows:

// I started using CMSSW_6_1_0_pre7 with this command:
// runTheMatrix.py -l 24.0
// Then I reran step3 with the the last line of the following code
// fragment added in ProductRegistry.cc to print out the products
// as they are added to the lookup table.
/*
  void ProductRegistry::initializeLookupTables() const {

    ...

          fillLookup(type, index, pBD, tempProductLookupMap);
          std::cout << "initializeLookups: " << desc.className() << "\n"
                    << "initializeLookups: " << desc.moduleLabel() << "\n" 
                    << "initializeLookups: " << desc.productInstanceName() << "\n"
                    << "initializeLookups: " << desc.processName() << "\n"; 
*/

// Then run  it with this command
//   cmsRun step3_RAW2DIGI_L1Reco_RECO_VALIDATION_DQM.py > & log2
// Then grep out the interesting lines with this command
//   grep initializeLook log2 > log3

// Just running the test will print the important timing info
// to std::cout. In addition if you want to run the program
// under igprof, use the following commands:
// igprof -d -pp -o igprof.pp.gz ../tmp/slc5_amd64_gcc472/src/DataFormats/Provenance/test/productResolverIndexHelperTest/productResolverIndexHelperTest > & logfile &
// igprof-analyse -d -v -g igprof.pp.gz > & igreport_perfres
// more igreport_perfres

using namespace edm;

namespace edmtestlookup {

  class Names {
  public:
    std::string className;
    std::string label;
    std::string instance;
    std::string process;
    TypeID typeID;
  };
}  // namespace edmtestlookup

using namespace edmtestlookup;

int main() {
  //Read the file listing all the products
  std::string line1;
  std::string line2;
  std::string line3;
  std::string line4;

  Names names;
  std::vector<Names> vNames;

  std::ifstream myfile("log3");
  if (myfile.is_open()) {
    while (myfile.good()) {
      getline(myfile, line1);
      if (!myfile.good())
        break;
      getline(myfile, line2);
      if (!myfile.good())
        break;
      getline(myfile, line3);
      if (!myfile.good())
        break;
      getline(myfile, line4);
      if (!myfile.good())
        break;

      std::istringstream ss1(line1);
      std::istringstream ss2(line2);
      std::istringstream ss3(line3);
      std::istringstream ss4(line4);

      std::string word;

      word = ss1.str();
      names.className = word.substr(19);
      ss2 >> word >> names.label;
      ss3 >> word >> names.instance;
      ss4 >> word >> names.process;

      // if (names.className == "trigger::TriggerFilterObjectWithRefs") continue;
      // if (vNames.size() > 99) continue;

      // std::cout << names.className << " " << names.label << " " << names.instance << " " << names.process << std::endl;

      vNames.push_back(names);
    }
    myfile.close();
  } else {
    std::cout << "ERROR: could not open file log3\n";
  }

  std::cout << "vNames.size = " << vNames.size() << "\n";

  edm::CPUTimer timer;

  timer.start();
  edm::ProductResolverIndexHelper phih;
  timer.stop();

  std::vector<unsigned int> savedIndexes;

  for (auto& n : vNames) {
    // Most of the initialization time in this test is
    // in this line as the dictionaries are loaded. Depending
    // on the number of loops and types, this also may be
    // most of the total time as well.
    TClass* clss = TClass::GetClass(n.className.c_str());

    if (n.className == "bool") {
      n.typeID = TypeID(typeid(bool));
    } else if (n.className == "double") {
      n.typeID = TypeID(typeid(double));
    } else {
      n.typeID = TypeID(*clss->GetTypeInfo());
    }

    timer.start();
    phih.insert(n.typeID, n.label.c_str(), n.instance.c_str(), n.process.c_str());
    timer.stop();
  }
  std::cout << "Filling Time: real " << timer.realTime() << " cpu " << timer.cpuTime() << std::endl;
  timer.reset();

  timer.start();
  phih.setFrozen();
  timer.stop();
  std::cout << "Freezing Time: real " << timer.realTime() << " cpu " << timer.cpuTime() << std::endl;
  timer.reset();

  // phih.print(std::cout);

  timer.start();

  unsigned sum = 0;

  for (unsigned j = 0; j < 100; ++j) {
    for (auto& n : vNames) {
      unsigned temp = 1;
      // if (n.className == "trigger::TriggerFilterObjectWithRefs") continue;
      temp = phih.index(PRODUCT_TYPE, n.typeID, n.label.c_str(), n.instance.c_str(), n.process.c_str());

      sum += temp;
    }
  }
  timer.stop();

  std::cout << "index loop time = real " << timer.realTime() << " cpu " << timer.cpuTime() << std::endl;
  return sum;
}
