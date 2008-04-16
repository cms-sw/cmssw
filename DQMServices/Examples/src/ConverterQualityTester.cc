#include "DQMServices/Examples/interface/ConverterQualityTester.h"

ConverterQualityTester::ConverterQualityTester(const edm::ParameterSet& iPSet)
{
  std::string MsgLoggerCat = "ConverterQualityTester_ConverterQualityTester";

  fName = iPSet.getUntrackedParameter<std::string>("Name");
  verbosity = iPSet.getUntrackedParameter<int>("Verbosity");
 
  if (verbosity >= 0) {
    edm::LogInfo(MsgLoggerCat) 
      << "\n===============================\n"
      << "Initialized as EDAnalyzer with parameter values:\n"
      << "    Name          = " << fName << "\n"
      << "    Verbosity     = " << verbosity << "\n"
      << "===============================\n";
  }
}

ConverterQualityTester::~ConverterQualityTester() {}

void ConverterQualityTester::beginJob(const edm::EventSetup& iSetup)
{
  return;
}

void ConverterQualityTester::endJob()
{
  return;
}

void ConverterQualityTester::beginRun(const edm::Run& iRun, 
				const edm::EventSetup& iSetup)
{
  return;
}

void ConverterQualityTester::endRun(const edm::Run& iRun, 
			      const edm::EventSetup& iSetup)
{
  return;
}

void ConverterQualityTester::analyze(const edm::Event& iEvent, 
			       const edm::EventSetup& iSetup)
{
  return;
}
