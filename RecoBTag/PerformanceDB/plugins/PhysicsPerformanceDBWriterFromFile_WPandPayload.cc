#include <memory>
#include <string>
#include <fstream>
#include <iostream>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/PhysicsToolsObjects/interface/PerformancePayloadFromTable.h"

#include "CondFormats/PhysicsToolsObjects/interface/PerformanceWorkingPoint.h"

class PhysicsPerformanceDBWriterFromFile_WPandPayload : public edm::EDAnalyzer {
public:
  PhysicsPerformanceDBWriterFromFile_WPandPayload(const edm::ParameterSet&);
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override {}
  void endJob() override {}
  ~PhysicsPerformanceDBWriterFromFile_WPandPayload() override {}

private:
  std::string inputTxtFile;
  std::string rec1, rec2;
};

PhysicsPerformanceDBWriterFromFile_WPandPayload::PhysicsPerformanceDBWriterFromFile_WPandPayload(
    const edm::ParameterSet& p) {
  inputTxtFile = p.getUntrackedParameter<std::string>("inputTxtFile");
  rec1 = p.getUntrackedParameter<std::string>("RecordPayload");
  rec2 = p.getUntrackedParameter<std::string>("RecordWP");
}

void PhysicsPerformanceDBWriterFromFile_WPandPayload::beginJob() {
  //
  // read object from file
  //

  //
  // File Format is
  // - tagger name
  // - cut
  // - concrete class name
  // - how many results and how many binning
  // - their values
  // - vector<float>
  //

  std::ifstream in;
  edm::LogInfo("PhysicsPerformanceDBWriterFromFile_WPandPayload") << "Opening " << inputTxtFile;
  in.open(inputTxtFile.c_str());
  std::string tagger;
  float cut;

  std::string concreteType;
  std::string comment;
  std::vector<float> pl;
  int stride;

  in >> tagger;
  edm::LogInfo("PhysicsPerformanceDBWriterFromFile_WPandPayload") << "WP Tagger is " << tagger;

  in >> cut;
  edm::LogInfo("PhysicsPerformanceDBWriterFromFile_WPandPayload") << "WP Cut is " << cut;

  in >> concreteType;
  edm::LogInfo("PhysicsPerformanceDBWriterFromFile_WPandPayload") << "concrete Type is " << concreteType;

  //  return ;

  // read # of results

  int nres, nbin;
  in >> nres;
  in >> nbin;
  edm::LogInfo("PhysicsPerformanceDBWriterFromFile_WPandPayload")
      << " Results: " << nres << " Binning variables: " << nbin;

  stride = nres + nbin * 2;
  if (!stride) {
    edm::LogInfo("PhysicsPerformanceDBWriterFromFile_WPandPayload") << " Malformed input file";
    exit(1);
  }

  int number = 0;

  std::vector<PerformanceResult::ResultType> res;
  std::vector<BinningVariables::BinningVariablesType> bin;

  while (number < nres && !in.eof()) {
    int tmp;
    in >> tmp;
    res.push_back((PerformanceResult::ResultType)(tmp));
    number++;
  }
  if (number != nres) {
    edm::LogInfo("PhysicsPerformanceDBWriterFromFile_WPandPayload") << "Table not well formed";
  }
  number = 0;
  while (number < nbin && !in.eof()) {
    int tmp;
    in >> tmp;
    bin.push_back((BinningVariables::BinningVariablesType)(tmp));
    number++;
  }
  if (number != nbin) {
    edm::LogInfo("PhysicsPerformanceDBWriterFromFile_WPandPayload") << "Table not well formed";
  }

  number = 0;
  while (!in.eof()) {
    float temp;
    in >> temp;
    edm::LogInfo("PhysicsPerformanceDBWriterFromFile_WPandPayload")
        << " Intersing " << temp << " in position " << number;
    number++;
    pl.push_back(temp);
  }

  //
  // CHECKS
  //
  if (stride != nbin * 2 + nres) {
    edm::LogInfo("PhysicsPerformanceDBWriterFromFile_WPandPayload") << "Table not well formed";
  }
  if ((number % stride) != 0) {
    edm::LogInfo("PhysicsPerformanceDBWriterFromFile_WPandPayload") << "Table not well formed";
  }
  edm::LogInfo("PhysicsPerformanceDBWriterFromFile_WPandPayload") << " CLOSING ";
  in.close();

  /*  for (int k=0;k<(number/stride); k++){
    for (int j=0; j<stride; j++){
      std::cout << "Pos["<<k<<","<<j<<"] = "<<pl[k*stride+j]<<std::endl;
    }
  }
  */

  //
  // now create pl etc etc
  //

  PerformanceWorkingPoint wp(cut, tagger);

  PerformancePayloadFromTable btagpl;

  if (concreteType == "PerformancePayloadFromTable") {
    btagpl = PerformancePayloadFromTable(res, bin, stride, pl);
  } else {
    edm::LogInfo("PhysicsPerformanceDBWriterFromFile_WPandPayload") << " Non existing request: " << concreteType;
  }

  edm::LogInfo("PhysicsPerformanceDBWriterFromFile_WPandPayload") << " Created the " << concreteType << " object";

  edm::Service<cond::service::PoolDBOutputService> s;
  if (s.isAvailable()) {
    s->writeOneIOV(btagpl, s->beginOfTime(), rec1);
    // write also the WP
    s->writeOneIOV(wp, s->beginOfTime(), rec2);
  }
}

DEFINE_FWK_MODULE(PhysicsPerformanceDBWriterFromFile_WPandPayload);
