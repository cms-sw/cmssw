#include "DQMServices/Examples/interface/ConverterTester.h"

ConverterTester::ConverterTester(const edm::ParameterSet& iPSet)
{
  std::string MsgLoggerCat = "ConverterTester_ConverterTester";

  fName = iPSet.getUntrackedParameter<std::string>("Name");
  verbosity = iPSet.getUntrackedParameter<int>("Verbosity");
  frequency = iPSet.getUntrackedParameter<int>("Frequency");
 
  if (verbosity >= 0) {
    edm::LogInfo(MsgLoggerCat) 
      << "\n===============================\n"
      << "Initialized as EDAnalyzer with parameter values:\n"
      << "    Name          = " << fName << "\n"
      << "    Verbosity     = " << verbosity << "\n"
      << "    Frequency     = " << frequency << "\n"
      << "===============================\n";
  }
 
  dbe = 0;
  dbe = edm::Service<DQMStore>().operator->();
 
  count = 0;

}

ConverterTester::~ConverterTester() {}

void ConverterTester::beginJob()
{

  if(dbe){
    meTestString = 0;
    meTestInt = 0;
    meTestFloat = 0;
    meTestTH1FN = 0;
    meTestTH1FD = 0;
    meTestTH2F = 0;
    meTestTH3F = 0;
    meTestProfile1 = 0;
    meTestProfile2 = 0;
    Random = new TRandom3();
    
    dbe->setCurrentFolder("ConverterTest/String");
    meTestString = dbe->bookString("TestString","Test String" );

    dbe->setCurrentFolder("ConverterTest/Int");
    meTestInt = dbe->bookInt("TestInt");

    dbe->setCurrentFolder("ConverterTest/Float");
    meTestFloat = dbe->bookFloat("TestFloat");

    dbe->setCurrentFolder("ConverterTest/TH1F");
    meTestTH1FN = 
      dbe->book1D("Random1DN", "Random1D Numerator", 100, -10., 10.);
    meTestTH1FD = 
      dbe->book1D("Random1DD", "Random1D Denominator", 100, -10., 10.);

    dbe->setCurrentFolder("ConverterTest/TH2F");
    meTestTH2F = dbe->book2D("Random2D", "Random2D", 100, -10, 10., 100, -10., 
			     10.);

    dbe->setCurrentFolder("ConverterTest/TH3F");
    meTestTH3F = dbe->book3D("Random3D", "Random3D", 100, -10., 10., 100, 
			     -10., 10., 100, -10., 10.);

    dbe->setCurrentFolder("ConverterTest/TProfile");
    meTestProfile1 = dbe->bookProfile("Profile1", "Profile1", 100, -10., 10., 
				      100, -10., 10.);

    dbe->setCurrentFolder("ConverterTest/TProfile2D");
    meTestProfile2 = dbe->bookProfile2D("Profile2", "Profile2", 100, -10., 
					10., 100, -10, 10., 100, -10., 10.);

    dbe->tag(meTestTH1FN->getFullname(),1);
    dbe->tag(meTestTH1FD->getFullname(),2);    
    dbe->tag(meTestTH2F->getFullname(),3);
    dbe->tag(meTestTH3F->getFullname(),4);
    dbe->tag(meTestProfile1->getFullname(),5);
    dbe->tag(meTestProfile2->getFullname(),6);
    dbe->tag(meTestString->getFullname(),7);
    dbe->tag(meTestInt->getFullname(),8);
    dbe->tag(meTestFloat->getFullname(),9);
  }

  return;
}

void ConverterTester::endJob()
{
  std::string MsgLoggerCat = "ConverterTester_endJob";
  if (verbosity >= 0)
    edm::LogInfo(MsgLoggerCat) 
      << "Terminating having processed " << count << " events.";
  return;
}

void ConverterTester::beginRun(const edm::Run& iRun, 
				const edm::EventSetup& iSetup)
{
  return;
}

void ConverterTester::endRun(const edm::Run& iRun, 
			      const edm::EventSetup& iSetup)
{
  meTestInt->Fill(100);
  meTestFloat->Fill(3.141592);

  return;
}

void ConverterTester::analyze(const edm::Event& iEvent, 
			       const edm::EventSetup& iSetup)
{
  
  ++count;

  for(int i = 0; i < 1000; ++i) {
    RandomVal1 = Random->Gaus(0.,1.);
    RandomVal2 = Random->Gaus(0.,1.);
    RandomVal3 = Random->Gaus(0.,1.);
    
    meTestTH1FN->Fill(0.5*RandomVal1);
    meTestTH1FD->Fill(RandomVal1);    
    meTestTH2F->Fill(RandomVal1, RandomVal2);
    meTestTH3F->Fill(RandomVal1, RandomVal2, RandomVal3);
    meTestProfile1->Fill(RandomVal1, RandomVal2);
    meTestProfile2->Fill(RandomVal1, RandomVal2, RandomVal3);
  }
}
