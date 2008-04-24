#include <ostream>
#include <iostream>
#include <string>
#include <vector>

#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/parse.h"
#include "FWCore/ParameterSet/interface/ParseTree.h"
#include "FWCore/ParameterSet/src/PythonFormWriter.h"

using namespace edm;
using namespace edm::pset;

void
writePythonForm(std::string const& config, std::ostream& out)
{
  edm::pset::ParseTree parsetree(config);
  
  PythonFormWriter writer;
  writer.write(parsetree, out);
}


int main()
{
  // Read input from cin into configstring..
  std::string configstring;
  edm::read_from_cin(configstring);

  // Now parse this configuration string, writing the Python format to
  // standard out.

  int rc = 1;  // failure
  try  
    { 
      writePythonForm(configstring, std::cout);
      rc = 0; // success
    }
  catch ( edm::Exception const& x )
    {
      std::cerr << x << '\n';
    }
  catch ( ... )
    {
      std::cerr << "Unidentified exception caught\n";	
    }
  return rc;  
}
