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
  std::cout << "Opening " << inputTxtFile << std::endl;
  in.open(inputTxtFile.c_str());
  std::string tagger;
  float cut;

  std::string concreteType;
  std::string comment;
  std::vector<float> pl;
  int stride;

  in >> tagger;
  std::cout << "WP Tagger is " << tagger << std::endl;

  in >> cut;
  std::cout << "WP Cut is " << cut << std::endl;

  in >> concreteType;
  std::cout << "concrete Type is " << concreteType << std::endl;

  //  return ;

  // read # of results

  int nres, nbin;
  in >> nres;
  in >> nbin;
  std::cout << " Results: " << nres << " Binning variables: " << nbin << std::endl;

  stride = nres + nbin * 2;
  if (!stride) {
    std::cout << " Malformed input file" << std::endl;
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
    std::cout << " Table not well formed" << std::endl;
  }
  number = 0;
  while (number < nbin && !in.eof()) {
    int tmp;
    in >> tmp;
    bin.push_back((BinningVariables::BinningVariablesType)(tmp));
    number++;
  }
  if (number != nbin) {
    std::cout << " Table not well formed" << std::endl;
  }

  number = 0;
  while (!in.eof()) {
    float temp;
    in >> temp;
    std::cout << " Intersing " << temp << " in position " << number << std::endl;
    number++;
    pl.push_back(temp);
  }

  //
  // CHECKS
  //
  if (stride != nbin * 2 + nres) {
    std::cout << " Table not well formed" << std::endl;
  }
  if ((number % stride) != 0) {
    std::cout << " Table not well formed" << std::endl;
  }

  std::cout << " CLOSING " << std::endl;
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
    btagpl= PerformancePayloadFromTable(res, bin, stride, pl);
  } else {
    std::cout << " Non existing request: " << concreteType << std::endl;
  }

  std::cout << " Created the " << concreteType << " object" << std::endl;

  edm::Service<cond::service::PoolDBOutputService> s;
  if (s.isAvailable()) {
    s->writeOneIOV(btagpl, s->beginOfTime(), rec1);
    // write also the WP
    s->writeOneIOV(wp, s->beginOfTime(), rec2);
  }
}

DEFINE_FWK_MODULE(PhysicsPerformanceDBWriterFromFile_WPandPayload);
