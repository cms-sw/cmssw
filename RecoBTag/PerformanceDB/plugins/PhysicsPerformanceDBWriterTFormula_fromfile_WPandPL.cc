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
#include "CondFormats/PhysicsToolsObjects/interface/PerformancePayloadFromBinnedTFormula.h"
#include "CondFormats/PhysicsToolsObjects/interface/PerformanceWorkingPoint.h"

class PhysicsPerformanceDBWriterTFormula_fromfile_WPandPL : public edm::global::EDAnalyzer<> {
public:
  PhysicsPerformanceDBWriterTFormula_fromfile_WPandPL(const edm::ParameterSet&);
  void beginJob() override;
  void analyze(edm::StreamID, const edm::Event&, const edm::EventSetup&) const override {}
  void endJob() override {}
  ~PhysicsPerformanceDBWriterTFormula_fromfile_WPandPL() override {}

private:
  std::string inputTxtFile;
  std::string rec1, rec2;
};

PhysicsPerformanceDBWriterTFormula_fromfile_WPandPL::PhysicsPerformanceDBWriterTFormula_fromfile_WPandPL(
    const edm::ParameterSet& p) {
  inputTxtFile = p.getUntrackedParameter<std::string>("inputTxtFile");
  rec1 = p.getUntrackedParameter<std::string>("RecordPayload");
  rec2 = p.getUntrackedParameter<std::string>("RecordWP");
}

void PhysicsPerformanceDBWriterTFormula_fromfile_WPandPL::beginJob() {
  //
  // read object from file
  //

  //
  // File Format is
  // - tagger name
  // - cut
  // - concrete class name
  // number of results (btageff, btagSF....)
  // number of binning variables in the parameterization (eta, pt ...)
  // number of bins
  // - results (as ints)
  // - variables (as ints)
  // - formulas
  // - the limits
  //

  std::ifstream in;
  in.open(inputTxtFile.c_str());
  std::string tagger;
  float cut;

  std::string concreteType;

  in >> tagger;
  edm::LogInfo("PhysicsPerformanceDBWriterTFormula_fromfile_WPandPL") << "WP Tagger is " << tagger;

  in >> cut;
  edm::LogInfo("PhysicsPerformanceDBWriterTFormula_fromfile_WPandPL") << "WP Cut is " << cut;

  in >> concreteType;
  edm::LogInfo("PhysicsPerformanceDBWriterTFormula_fromfile_WPandPL") << "concrete Type is " << concreteType;

  int nres = 0, nvar = 0;

  in >> nres;
  in >> nvar;

  edm::LogInfo("PhysicsPerformanceDBWriterTFormula_fromfile_WPandPL")
      << "Using " << nres << " results and " << nvar << " variables";

  unsigned int bins = 0;  //temporary for now!!!!!!

  in >> bins;

  edm::LogInfo("PhysicsPerformanceDBWriterTFormula_fromfile_WPandPL") << "Using " << bins << " bins";

  int number = 0;
  ;

  std::vector<PerformanceResult::ResultType> res;
  std::vector<BinningVariables::BinningVariablesType> bin;
  //
  // read results
  //
  number = 0;
  while (number < nres && !in.eof()) {
    int tmp;
    in >> tmp;
    res.push_back((PerformanceResult::ResultType)(tmp));
    edm::LogInfo("PhysicsPerformanceDBWriterTFormula_fromfile_WPandPL") << " Result #" << number << " is " << tmp;
    number++;
  }
  if (number != nres) {
    edm::LogInfo("PhysicsPerformanceDBWriterTFormula_fromfile_WPandPL") << " Table not well formed";
  }

  //
  // read the variables
  //

  PerformanceWorkingPoint wp(cut, tagger);
  PerformancePayloadFromBinnedTFormula btagpl;

  std::vector<PhysicsTFormulaPayload> v_ppl;

  number = 0;
  while (number < nvar && !in.eof()) {
    int tmp;
    in >> tmp;
    bin.push_back((BinningVariables::BinningVariablesType)(tmp));
    edm::LogInfo("PhysicsPerformanceDBWriterTFormula_fromfile_WPandPL") << " Variable #" << number << " is " << tmp;
    number++;
  }
  if (number != nvar) {
    edm::LogInfo("PhysicsPerformanceDBWriterTFormula_fromfile_WPandPL") << " Table not well formed";
  }

  //
  // now read the formulas
  //

  for (unsigned int recregion = 0; recregion < bins; ++recregion) {
    std::vector<std::pair<float, float> > limits;
    std::vector<std::string> formulas;

    number = 0;

    while (number < nres && (!in.eof())) {
      std::string temp;
      in >> temp;
      edm::LogInfo("PhysicsPerformanceDBWriterTFormula_fromfile_WPandPL")
          << " Inserting " << temp << " as formula in position " << number;
      number++;
      formulas.push_back(temp);
    }
    /*
      if (nres!= number ){
      std::cout <<" NOT OK, this is not what I would expect"<<std::endl;
      abort();
      }
    */

    number = 0;
    while (number < nvar && (!in.eof())) {
      float temp1, temp2;
      in >> temp1;
      in >> temp2;
      edm::LogInfo("PhysicsPerformanceDBWriterTFormula_fromfile_WPandPL")
          << " Inserting " << temp1 << "," << temp2 << " as limits in position " << number;
      number++;
      limits.push_back(std::pair<float, float>(temp1, temp2));
    }
    /*
      if (nvar != number ){
      std::cout <<" NOT OK, this is not what I would expect"<<std::endl;
      abort();
      }
    */

    //
    // push it
    //

    PhysicsTFormulaPayload ppl(limits, formulas);
    v_ppl.push_back(ppl);
  }
  in.close();

  if (concreteType == "PerformancePayloadFromBinnedTFormula") {
    btagpl = PerformancePayloadFromBinnedTFormula(res, bin, v_ppl);
    edm::LogInfo("PhysicsPerformanceDBWriterTFormula_fromfile_WPandPL")
        << " CHECK: " << btagpl.formulaPayloads().size();
  } else {
    edm::LogInfo("PhysicsPerformanceDBWriterTFormula_fromfile_WPandPL") << " Non existing request: " << concreteType;
  }

  edm::LogInfo("PhysicsPerformanceDBWriterTFormula_fromfile_WPandPL") << " Created the " << concreteType << " object";

  edm::LogInfo("PhysicsPerformanceDBWriterTFormula_fromfile_WPandPL") << "Start writing the payload and WP";
  edm::Service<cond::service::PoolDBOutputService> s;
  if (s.isAvailable()) {
    s->writeOneIOV(btagpl, s->beginOfTime(), rec1);
    // write also the WP
    s->writeOneIOV(wp, s->beginOfTime(), rec2);
  }

  edm::LogInfo("PhysicsPerformanceDBWriterTFormula_fromfile_WPandPL") << "Finised writing the payload and WP";
}

DEFINE_FWK_MODULE(PhysicsPerformanceDBWriterTFormula_fromfile_WPandPL);
