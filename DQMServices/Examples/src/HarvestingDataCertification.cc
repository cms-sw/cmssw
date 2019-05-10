#include "DQMServices/Examples/interface/HarvestingDataCertification.h"

HarvestingDataCertification::HarvestingDataCertification(const edm::ParameterSet &iPSet) {
  std::string MsgLoggerCat = "HarvestingDataCertification_HarvestingDataCertification";

  fName = iPSet.getUntrackedParameter<std::string>("Name");
  verbosity = iPSet.getUntrackedParameter<int>("Verbosity");

  if (verbosity >= 0) {
    edm::LogInfo(MsgLoggerCat) << "\n===============================\n"
                               << "Initialized as EDAnalyzer with parameter values:\n"
                               << "    Name          = " << fName << "\n"
                               << "    Verbosity     = " << verbosity << "\n"
                               << "===============================\n";
  }
}

HarvestingDataCertification::~HarvestingDataCertification() {}

void HarvestingDataCertification::beginJob() { return; }

void HarvestingDataCertification::endJob() { return; }

void HarvestingDataCertification::beginRun(const edm::Run &iRun, const edm::EventSetup &iSetup) { return; }

void HarvestingDataCertification::endRun(const edm::Run &iRun, const edm::EventSetup &iSetup) { return; }

void HarvestingDataCertification::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) { return; }
