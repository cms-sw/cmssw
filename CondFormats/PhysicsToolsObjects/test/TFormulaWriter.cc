#include "CondFormats/PhysicsToolsObjects/test/TFormulaWriter.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "CondFormats/PhysicsToolsObjects/interface/PhysicsTFormulaPayload.h"

#include <TFile.h>
#include <TFormula.h>

TFormulaWriter::TFormulaWriter(const edm::ParameterSet& cfg)
    : moduleLabel_(cfg.getParameter<std::string>("@module_label")) {
  edm::VParameterSet cfgJobs = cfg.getParameter<edm::VParameterSet>("jobs");
  for (const auto& cfgJob : cfgJobs) {
    jobEntryType* job = new jobEntryType(cfgJob);
    jobs_.push_back(job);
  }
}

TFormulaWriter::~TFormulaWriter() {
  for (auto& job : jobs_) {
    delete job;
  }
}

void TFormulaWriter::analyze(const edm::Event&, const edm::EventSetup&) {
  std::cout << "<TFormulaWriter::analyze (moduleLabel = " << moduleLabel_ << ")>:" << std::endl;

  for (auto& job : jobs_) {
    TFile* inputFile = new TFile(job->inputFileName_.data());
    std::cout << "reading TFormula = " << job->formulaName_ << " from ROOT file = " << job->inputFileName_ << "."
              << std::endl;
    const TFormula* formula = dynamic_cast<TFormula*>(inputFile->Get(job->formulaName_.data()));
    std::cout << "the formula is " << formula->GetExpFormula("p") << std::endl;
    delete inputFile;
    if (!formula)
      throw cms::Exception("TFormulaWriter") << " Failed to load TFormula = " << job->formulaName_.data()
                                             << " from file = " << job->inputFileName_ << " !!\n";
    edm::Service<cond::service::PoolDBOutputService> dbService;
    if (!dbService.isAvailable())
      throw cms::Exception("TFormulaWriter") << " Failed to access PoolDBOutputService !!\n";
    std::cout << " writing TFormula = " << job->formulaName_ << " to SQLlite file, record = " << job->outputRecord_
              << "." << std::endl;
    typedef std::pair<float, float> vfloat;
    std::vector<vfloat> limits;
    limits.emplace_back(0., 1.e+6);
    std::vector<std::string> formulas;
    formulas.emplace_back((formula->GetExpFormula("p")).Data());
    auto* formulaPayload = new PhysicsTFormulaPayload(limits, formulas);
    delete formula;
    dbService->writeOne(formulaPayload, dbService->beginOfTime(), job->outputRecord_);
  }

  std::cout << "done." << std::endl;
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(TFormulaWriter);
