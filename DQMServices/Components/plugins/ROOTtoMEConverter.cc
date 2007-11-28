/** \file ROOTtoMEConverter.cc
 *  
 *  See header file for description of class
 *
 *  $Date: 2007/11/20 23:53:45 $
 *  $Revision: 1.3 $
 *  \author M. Strang SUNY-Buffalo
 */

#include "DQMServices/Components/plugins/ROOTtoMEConverter.h"

ROOTtoMEConverter::ROOTtoMEConverter(const edm::ParameterSet & iPSet) :
  fName(""), verbosity(0), frequency(0), count(0)
{
  std::string MsgLoggerCat = "ROOTtoMEConverter_ROOTtoMEConverter";
  
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
      << "Initialized as EDAnalyzer with parameter values:\n"
      << "    Name          = " << fName << "\n"
      << "    Verbosity     = " << verbosity << "\n"
      << "    Frequency     = " << frequency << "\n"
      << "===============================\n";
  }
  
  // create persistent object(s)
  // produces<TYPE, edm::InRun>(NAME).setBranchAlias(NAME2);
  
} // end constructor

ROOTtoMEConverter::~ROOTtoMEConverter() 
{
} // end destructor

void ROOTtoMEConverter::beginJob(const edm::EventSetup& iSetup)
{
  return;
}

void ROOTtoMEConverter::endJob()
{
  std::string MsgLoggerCat = "ROOTtoMEConverter_endJob";
  if (verbosity >= 0)
    edm::LogInfo(MsgLoggerCat) 
      << "Terminating having processed " << count << " runs.";
  return;
}

void ROOTtoMEConverter::beginRun(const edm::Run& iRun, 
				 const edm::EventSetup& iSetup)
{
  std::string MsgLoggerCat = "ROOTtoMEConverter_beginRun";
  
  // keep track of number of events processed
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

void ROOTtoMEConverter::endRun(const edm::Run& iRun, 
			       const edm::EventSetup& iSetup)
{
  
  std::string MsgLoggerCat = "ROOTtoMEConverter_endRun";
  
  if (verbosity > 0)
    edm::LogInfo (MsgLoggerCat)
      << "\nRestoring MonitorElements.";
  
  // do stuff here

  return;
}

void ROOTtoMEConverter::analyze(const edm::Event& iEvent, 
				const edm::EventSetup& iSetup)
{
  return;
}
