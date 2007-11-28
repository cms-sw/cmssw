/** \file MEtoROOTConverter.cc
 *  
 *  See header file for description of class
 *
 *  $Date: 2007/11/20 23:53:45 $
 *  $Revision: 1.3 $
 *  \author M. Strang SUNY-Buffalo
 */

#include "DQMServices/Components/plugins/MEtoROOTConverter.h"

MEtoROOTConverter::MEtoROOTConverter(const edm::ParameterSet & iPSet) :
  fName(""), verbosity(0), frequency(0), count(0)
{
  std::string MsgLoggerCat = "MEtoROOTConverter_MEtoROOTConverter";

  // get information from parameter set
  fName = iPSet.getUntrackedParameter<std::string>("Name");
  verbosity = iPSet.getUntrackedParameter<int>("Verbosity");
  frequency = iPSet.getUntrackedParameter<int>("Frequency");
  
  // use value of first digit to determine default output level (inclusive)
  // 0 is none, 1 is basic, 2 is fill output, 3 is gather output
  verbosity %= 10;
  
  // print out Parameter Set information being used
  if (verbosity >= 0) {
    edm::LogInfo(MsgLoggerCat) 
      << "\n===============================\n"
      << "Initialized as EDProducer with parameter values:\n"
      << "    Name          = " << fName << "\n"
      << "    Verbosity     = " << verbosity << "\n"
      << "    Frequency     = " << frequency << "\n"
      << "===============================\n";
  }
  
  // get dqm info
  dbe = 0;
  dbe = edm::Service<DaqMonitorBEInterface>().operator->();
  if (dbe) {
    if (verbosity >= 0 ) {
      dbe->setVerbose(1);
    } else {
      dbe->setVerbose(0);
    }
  }

  //if (dbe) {
  //  if (verbosity >= 0 ) dbe->showDirStructure();
  //}



  // create persistent object(s)
  produces<MEtoROOT, edm::InRun>("test").setBranchAlias("test");
  
} // end constructor

MEtoROOTConverter::~MEtoROOTConverter() 
{
} // end destructor

void MEtoROOTConverter::beginJob(const edm::EventSetup& iSetup)
{
  return;
}

void MEtoROOTConverter::endJob()
{
  std::string MsgLoggerCat = "MEtoROOTConverter_endJob";
  if (verbosity >= 0)
    edm::LogInfo(MsgLoggerCat) 
      << "Terminating having processed " << count << " runs.";
  return;
}

void MEtoROOTConverter::beginRun(edm::Run& iRun, 
				 const edm::EventSetup& iSetup)
{
  std::string MsgLoggerCat = "MEtoROOTConverter_beginRun";
  
  // keep track of number of runs processed
  ++count;
  
  int nrun = iRun.run();
  
  if (verbosity > 0) {
    edm::LogInfo(MsgLoggerCat)
      << "Processing run " << nrun << " (" << count << " runs total)";
  } else if (verbosity == 0) {
    if (nrun%frequency == 0 || nrun == 0) {
      edm::LogInfo(MsgLoggerCat)
	<< "Processing run " << nrun << " (" << count << " runs total)";
    }
  }
  
  return;
}

void MEtoROOTConverter::endRun(edm::Run& iRun, const edm::EventSetup& iSetup)
{
  
  std::string MsgLoggerCat = "MEtoROOTConverter_endRun";
  
  if (verbosity >= 0)
    edm::LogInfo (MsgLoggerCat)
      << "\nStoring MEtoROOT dataformat histograms.";
  
  // do stuff here

  return;
}

void MEtoROOTConverter::produce(edm::Event& iEvent, 
				const edm::EventSetup& iSetup)
{
  return;
}
