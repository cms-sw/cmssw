#include <iostream>
#include <sstream>
#include <string>
#include <vector>


#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Services/src/JobReportService.h"

void work()
{
  // Make the service.
  edm::ParameterSet ps;
  edm::ActivityRegistry areg;

  edm::service::JobReportService jrs(ps, areg);
  std::vector<edm::JobReport::Token> inputTokens;
  std::vector<edm::JobReport::Token> outputTokens;

  // Make input files
  std::vector<std::string> branchnames;
  branchnames.push_back("branch_1_");
  branchnames.push_back("branch_2_");

  for (int i = 0; i < 10; ++i) {
      std::ostringstream oss;
      oss << i;
      std::string seq_tag = oss.str();
      std::vector<std::string> localBranchNames(branchnames);
      localBranchNames[0] += seq_tag;
      localBranchNames[1] += seq_tag;
      edm::JobReport::Token t = jrs.inputFileOpened("phys" + seq_tag,
				    "log" + seq_tag,
				    "cat" + seq_tag,
				    "class" + seq_tag,
				    "label" + seq_tag,
				    localBranchNames);
      inputTokens.push_back(t);
  }


  jrs.reportSkippedEvent(10001, 1002);
  jrs.reportSkippedEvent(10001, 1003);
  jrs.reportSkippedEvent(10001, 1004);

  try {
      jrs.eventReadFromFile(24, 0, 0);
      assert( "Failed to throw required exception" == 0);
  }
  catch ( edm::Exception & x ) {
      assert( x.categoryCode() == edm::errors::LogicError );
  }
  catch ( ... ) {
      assert( "Threw unexpected exception type" == 0 );
  }
  

  // Fake the end-of-job
  areg.postEndJobSignal_();
}

void fail()
{
  // Make the service.
  edm::ParameterSet ps;
  edm::ActivityRegistry areg;

  edm::service::JobReportService jrs(ps, areg);
  std::vector<edm::JobReport::Token> inputTokens;
  std::vector<edm::JobReport::Token> outputTokens;

  areg.jobFailureSignal_();
}

int main()
{
  int rc = -1;
  try {
      work();
      fail();
      rc = 0;
  }

  catch ( edm::Exception & x ) {
      std::cerr << x << '\n';
      rc = 1;
  }
  catch ( ... ) {
      std::cerr << "Unknown exception caught\n";
      rc = 2;
  }
  return rc;
}
