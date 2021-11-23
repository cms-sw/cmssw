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
#include "CondFormats/PhysicsToolsObjects/interface/PerformancePayloadFromBinnedTFormula.h"
#include "CondFormats/PhysicsToolsObjects/interface/PerformanceWorkingPoint.h"

class PhysicsPerformanceDBWriterTFormula_fromfile_WPandPL : public edm::EDAnalyzer {
public:
  PhysicsPerformanceDBWriterTFormula_fromfile_WPandPL(const edm::ParameterSet&);
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override {}
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
  edm::LogInfo("WP Tagger is ") << tagger << "\n";

  in >> cut;
  edm::LogInfo("WP Cut is ") << cut << "\n";

  in >> concreteType;
  edm::LogInfo("concrete Type is ") << concreteType << "\n";

  int nres = 0, nvar = 0;

  in >> nres;
  in >> nvar;

  edm::LogInfo("Using ") << nres << " results and " << nvar << " variables\n";

  unsigned int bins = 0;  //temporary for now!!!!!!

  in >> bins;

  edm::LogInfo("Using ") << bins << " bins\n";

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
    edm::LogInfo(" Result #") << number << " is " << tmp << "\n";
    number++;
  }
  if (number != nres) {
    edm::LogInfo(" Table not well formed\n");
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
    edm::LogInfo(" Variable #") << number << " is " << tmp << "\n";
    number++;
  }
  if (number != nvar) {
    edm::LogInfo(" Table not well formed\n");
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
      edm::LogInfo(" Inserting ") << temp << " as formula in position " << number << "\n";
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
      edm::LogInfo(" Inserting ") << temp1 << "," << temp2 << " as limits in position " << number << "\n";
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
    edm::LogInfo(" CHECK: ") << btagpl.formulaPayloads().size() << "\n";
  } else {
    edm::LogInfo(" Non existing request: ") << concreteType << "\n";
  }

  edm::LogInfo(" Created the ") << concreteType << " object\n";

  edm::LogInfo("Start writing the payload and WP\n");
  edm::Service<cond::service::PoolDBOutputService> s;
  if (s.isAvailable()) {
    s->writeOneIOV(btagpl, s->beginOfTime(), rec1);
    // write also the WP
    s->writeOneIOV(wp, s->beginOfTime(), rec2);
  }

  edm::LogInfo("Finised writing the payload and WP\n");
}

DEFINE_FWK_MODULE(PhysicsPerformanceDBWriterTFormula_fromfile_WPandPL);
