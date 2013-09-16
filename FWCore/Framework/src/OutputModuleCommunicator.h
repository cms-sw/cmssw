#ifndef FWCore_Framework_OutputModuleCommunicator_h
#define FWCore_Framework_OutputModuleCommunicator_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     OutputModuleCommunicator
// 
/**\class edm::OutputModuleCommunicator OutputModuleCommunicator.h "FWCore/Framework/interface/OutputModuleCommunicator.h"

 Description: Base class used by the framework to communicate with an OutputModule

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Fri, 05 Jul 2013 17:36:51 GMT
//

// system include files
#include <map>
#include <string>
#include <vector>

// user include files
#include "DataFormats/Provenance/interface/SelectedProducts.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

// forward declarations
namespace edm {

  class ProcessContext;

  class OutputModuleCommunicator
  {
    
  public:
    OutputModuleCommunicator() = default;
    virtual ~OutputModuleCommunicator();
    
    virtual void closeFile() = 0;
    
    ///\return true if output module wishes to close its file
    virtual bool shouldWeCloseFile() const = 0;
    
    virtual void openNewFileIfNeeded() = 0;
    
    ///\return true if no event filtering is applied to OutputModule
    virtual bool wantAllEvents() const = 0;
    
    virtual void openFile(FileBlock const& fb) = 0;
    
    virtual void writeRun(RunPrincipal const& rp, ProcessContext const*) = 0;
    
    virtual void writeLumi(LuminosityBlockPrincipal const& lbp, ProcessContext const*) = 0;
    
    ///\return true if OutputModule has reached its limit on maximum number of events it wants to see
    virtual bool limitReached() const = 0;
    
    virtual void configure(OutputModuleDescription const& desc) = 0;
    
    virtual SelectedProductsForBranchType const& keptProducts() const = 0;
    
    virtual void selectProducts(ProductRegistry const& preg) = 0;
    
    virtual void setEventSelectionInfo(std::map<std::string, std::vector<std::pair<std::string, int> > > const& outputModulePathPositions,
                                       bool anyProductProduced) = 0;

    virtual ModuleDescription const& description() const = 0;
    
  private:
    OutputModuleCommunicator(const OutputModuleCommunicator&) = delete; // stop default
    
    const OutputModuleCommunicator& operator=(const OutputModuleCommunicator&) = delete; // stop default
    
    // ---------- member data --------------------------------
    
  };
}

#endif
