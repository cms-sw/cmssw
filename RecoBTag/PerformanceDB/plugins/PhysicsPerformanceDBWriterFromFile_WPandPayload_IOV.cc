#include <memory>
#include <string>
#include <fstream>
#include <iostream>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/PhysicsToolsObjects/interface/PerformancePayloadFromTable.h"

#include "CondFormats/PhysicsToolsObjects/interface/PerformanceWorkingPoint.h"

class PhysicsPerformanceDBWriterFromFile_WPandPayload_IOV : public edm::global::EDAnalyzer<> {
public:
  PhysicsPerformanceDBWriterFromFile_WPandPayload_IOV(const edm::ParameterSet&);
  void beginJob() override;
  void analyze(edm::StreamID, const edm::Event&, const edm::EventSetup&) const override {}
  void endJob() override {}
  ~PhysicsPerformanceDBWriterFromFile_WPandPayload_IOV() override {}

private:
  std::string inputTxtFile;
  std::string rec1, rec2;
  unsigned long long iovBegin, iovEnd;
};

PhysicsPerformanceDBWriterFromFile_WPandPayload_IOV::PhysicsPerformanceDBWriterFromFile_WPandPayload_IOV(
    const edm::ParameterSet& p) {
  inputTxtFile = p.getUntrackedParameter<std::string>("inputTxtFile");
  rec1 = p.getUntrackedParameter<std::string>("RecordPayload");
  rec2 = p.getUntrackedParameter<std::string>("RecordWP");
  iovBegin = p.getParameter<unsigned long long>("IOVBegin");
  iovEnd = p.getParameter<unsigned long long>("IOVEnd");
}

void PhysicsPerformanceDBWriterFromFile_WPandPayload_IOV::beginJob() {
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
  in.open(inputTxtFile.c_str());
  std::string tagger;
  float cut;

  std::string concreteType;
  std::string comment;
  std::vector<float> pl;
  int stride;

  in >> tagger;
  edm::LogInfo("PhysicsPerformanceDBWriterFromFile_WPandPayload_IOV") << "WP Tagger is " << tagger;

  in >> cut;
  edm::LogInfo("PhysicsPerformanceDBWriterFromFile_WPandPayload_IOV") << "WP Cut is " << cut;

  in >> concreteType;
  edm::LogInfo("PhysicsPerformanceDBWriterFromFile_WPandPayload_IOV") << "concrete Type is " << concreteType;

  //  return ;

  // read # of results

  int nres, nbin;
  in >> nres;
  in >> nbin;
  edm::LogInfo("PhysicsPerformanceDBWriterFromFile_WPandPayload_IOV")
      << " Results: " << nres << " Binning variables: " << nbin;

  stride = nres + nbin * 2;

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
    edm::LogInfo("PhysicsPerformanceDBWriterFromFile_WPandPayload_IOV") << " Table not well formed";
  }
  number = 0;
  while (number < nbin && !in.eof()) {
    int tmp;
    in >> tmp;
    bin.push_back((BinningVariables::BinningVariablesType)(tmp));
    number++;
  }
  if (number != nbin) {
    edm::LogInfo("PhysicsPerformanceDBWriterFromFile_WPandPayload_IOV") << " Table not well formed";
  }

  number = 0;
  while (!in.eof()) {
    float temp;
    in >> temp;
    edm::LogInfo("PhysicsPerformanceDBWriterFromFile_WPandPayload_IOV")
        << " Inserting " << temp << " in position " << number;
    number++;
    pl.push_back(temp);
  }

  //
  // CHECKS
  //
  if (stride != nbin * 2 + nres) {
    edm::LogInfo("PhysicsPerformanceDBWriterFromFile_WPandPayload_IOV") << " Table not well formed";
  }
  if (stride != 0 && (number % stride) != 0) {
    edm::LogInfo("PhysicsPerformanceDBWriterFromFile_WPandPayload_IOV") << " Table not well formed";
  }

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
    edm::LogInfo("PhysicsPerformanceDBWriterFromFile_WPandPayload_IOV") << " Non existing request: " << concreteType;
  }

  edm::LogInfo("PhysicsPerformanceDBWriterFromFile_WPandPayload_IOV") << " Created the " << concreteType << " object";

  edm::Service<cond::service::PoolDBOutputService> s;
  if (s.isAvailable()) {
    s->writeOneIOV(btagpl, iovBegin, rec1);
    // write also the WP

    s->writeOneIOV(wp, iovBegin, rec2);
  }
}

DEFINE_FWK_MODULE(PhysicsPerformanceDBWriterFromFile_WPandPayload_IOV);
