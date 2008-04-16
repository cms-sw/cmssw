#include "DQMServices/Examples/interface/PostConverterAnalyzer.h"

PostConverterAnalyzer::PostConverterAnalyzer(const edm::ParameterSet& iPSet)
{
  std::string MsgLoggerCat = "PostConverterAnalyzer_PostConverterAnalyzer";

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

PostConverterAnalyzer::~PostConverterAnalyzer() {}

void PostConverterAnalyzer::beginJob(const edm::EventSetup& iSetup) 
{
  return;
}

void PostConverterAnalyzer::endJob()
{

  dbe = 0;
  dbe = edm::Service<DQMStore>().operator->();

  if (dbe) {

    // monitoring element numerator and denominator histogram
    MonitorElement *meN = 
      dbe->get("ConverterTest/TH1F/Random1DN");
    MonitorElement *meD = 
      dbe->get("ConverterTest/TH1F/Random1DD"); 

    if (meN && meD) {

      // get the numerator and denominator histogram    	
      TH1F *numerator = meN->getTH1F();
      numerator->Sumw2();
      TH1F *denominator = meD->getTH1F();
      denominator->Sumw2();

      // set the current directory
      dbe->setCurrentFolder("ConverterTest/TH1F");

      // booked the new histogram to contain the results
      MonitorElement *me = 
	dbe->book1D("Divide","Divide calculation",
		    numerator->GetNbinsX(),
		    numerator->GetXaxis()->GetXmin(),
		    numerator->GetXaxis()->GetXmax());

      // Calculate the efficiency
      me->getTH1F()->Divide(numerator, denominator, 1., 1., "B");

    } else {
      std::cout << "Monitor elements don't exist" << std::endl;
    }
  } else {
    std::cout << "Don't have a valid DQM back end" << std::endl;
  }

  return;
}

void PostConverterAnalyzer::beginRun(const edm::Run& iRun, 
				const edm::EventSetup& iSetup)
{
  return;
}

void PostConverterAnalyzer::endRun(const edm::Run& iRun, 
			      const edm::EventSetup& iSetup)
{
  return;
}

void PostConverterAnalyzer::analyze(const edm::Event& iEvent, 
			       const edm::EventSetup& iSetup)
{
  return;
}
