#ifndef FWCore_PythonParameterSet_MakeParameterSets_h
#define FWCore_PythonParameterSet_MakeParameterSets_h

//----------------------------------------------------------------------
// Declare functions used to create ParameterSets.
//----------------------------------------------------------------------

#include <memory>

#include <string>
#include <vector>

namespace edm {
  class ParameterSet;
  namespace  boost_python {
    
    // input can either be a python file name or a python config string
    std::unique_ptr<edm::ParameterSet>
      readConfig(std::string const& config);
    
    /// same, but with arguments
    std::unique_ptr<edm::ParameterSet>
      readConfig(std::string const& config, int argc, char* argv[]);
    
    /// essentially the same as the previous method
    void
      makeParameterSets(std::string const& configtext,
			std::unique_ptr<edm::ParameterSet> & main);
    
    /**finds all the PSets used in the top level module referred as a file or as a string containing
       python commands. These PSets are bundled into a top level PSet from which they can be retrieved
    */
    std::unique_ptr<edm::ParameterSet> readPSetsFrom(std::string const& fileOrString);
  } // BoostPython
} // namespace edm
#endif
  
