/*----------------------------------------------------------------------

This is a generic main that can be used with any plugin and a 
PSet script.   See notes in EventProcessor.cpp for details about
it.

$Id: cmsRun.cpp,v 1.2 2005/09/02 19:30:46 chrjones Exp $

----------------------------------------------------------------------*/  

#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "FWCore/Framework/interface/EventProcessor.h"
#include "FWCore/Utilities/interface/ProblemTracker.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace std;

// -----------------------------------------------

int main(int argc, char* argv[])
{
  // Right now option processing is not really present.
  // The EventProcessor should handle this in the future.

  edm::AssertHandler ah;

  int rc = -1; // we should never return this value!
  try
    {
      edm::EventProcessor proc(argc,argv);
      proc.beginJob();
      proc.run();
      if(proc.endJob()) {
        rc = 0;
      } else {
        rc = 1;
      }
    }
  catch (seal::Error& e)
    {
      std::cerr << "Exception caught in " << argv[0] << "\n"
		<< e.explainSelf()
		<< std::endl;
      rc = 1;
    }
  catch (std::exception& e)
    {
      std::cerr << "Standard library exception caught in " << argv[0] << "\n"
		<< e.what()
		<< std::endl;
      rc = 1;
    }
  catch (...)
    {
      std::cerr << "Unknown exception caught in " << argv[0]
		<< std::endl;
      rc = 2;
    }
  
  return rc;
}
