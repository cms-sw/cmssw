#ifndef FWCore_PyDevParameterSet_MakePyBind11ParameterSets_h
#define FWCore_PyDevParameterSet_MakePyBind11ParameterSets_h

//----------------------------------------------------------------------
// Declare functions used to create ParameterSets.
//----------------------------------------------------------------------

#include <memory>

#include <string>
#include <vector>

namespace edm {
  class ParameterSet;
  namespace cmspybind11_p3 {
    // input can either be a python file name or a python config string
    std::unique_ptr<ParameterSet> readConfig(std::string const& config);

    /// same, but with arguments
    std::unique_ptr<ParameterSet> readConfig(std::string const& config, int argc, char* argv[]);

    /// essentially the same as the previous method
    void makeParameterSets(std::string const& configtext, std::unique_ptr<ParameterSet>& main);

    /**finds all the PSets used in the top level module referred as a file or as a string containing
       python commands. These PSets are bundled into a top level PSet from which they can be retrieved
    */
    std::unique_ptr<ParameterSet> readPSetsFrom(std::string const& fileOrString);
  }  // namespace cmspybind11_p3
}  // namespace edm
#endif
