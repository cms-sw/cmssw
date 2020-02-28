#ifndef PhysicsTools_UtilAlgos_interface_FWLiteAnalyzerWrapper_h
#define PhysicsTools_UtilAlgos_interface_FWLiteAnalyzerWrapper_h

#include <string>
#include <vector>
#include <iostream>

#include <TFile.h>
#include <TSystem.h>

#include "DataFormats/FWLite/interface/ChainEvent.h"
#include "DataFormats/FWLite/interface/InputSource.h"
#include "DataFormats/FWLite/interface/OutputFiles.h"
#include "FWCore/FWLite/interface/FWLiteEnabler.h"
#include "FWCore/ParameterSet/interface/ProcessDesc.h"
#include "PhysicsTools/FWLite/interface/TFileService.h"

/**
  \class    FWLiteAnalyzerWrapper FWLiteAnalyzerWrapper.h "PhysicsTools/UtilAlgos/interface/FWLiteAnalyzerWrapper.h"
  \brief    Wrapper class for classes of type BasicAnalyzer to "convert" them into a full a basic FWLiteAnalyzer 

   This template class is a wrapper round classes of type BasicAnalyzer as defined in in the 
   BasicAnalyzer.h file of this package. From this class the wrapper expects the following 
   member functions:
   
   + a contructor with a const edm::ParameterSet& and a TFileDirectory& as input.
   + a beginJob function
   + a endJob function
   + a analyze function with an const edm::EventBase& as input
   
   these functions are called within the wrapper. The wrapper translates the common class into 
   a basic FWLiteAnalyzer, which provides basic functionality of reading configuration files 
   and event looping. An example of an application given in the PatExamples package is shown 
   below: 
   
   #include "PhysicsTools/PatExamples/interface/BasicMuonAnalyzer.h"
   #include "PhysicsTools/UtilAlgos/interface/FWLiteAnalyzerWrapper.h"
   
   typedef fwlite::AnalyzerWrapper<BasicMuonAnalyzer> WrappedFWLiteAnalyzer;
   
   int main(int argc, char* argv[]) 
   {
     // load framework libraries
     gSystem->Load( "libFWCoreFWLite" );
     FWLiteEnabler::enable();
     
     // only allow one argument for this simple example which should be the
     // the python cfg file
     if ( argc < 2 ) {
       std::cout << "Usage : " << argv[0] << " [parameters.py]" << std::endl;
       return 0;
     }

     // get the python configuration
     PythonProcessDesc builder(argv[1]);
     WrappedFWLiteAnalyzer ana(*(builder.processDesc()->getProcessPSet()), std::string("MuonAnalyzer"), std::string("analyzeBasicPat"));
     ana.beginJob();
     ana.analyze();
     ana.endJob();
     return 0;
   }

   The configuration file for this FWLiteAnalyzer is expected to have the following structure:

   import FWCore.ParameterSet.Config as cms
   
   process = cms.Process("FWLitePlots")

   process.fwliteInput = cms.PSet(
         fileNames = cms.untracked.vstring('file:patTuple.root'),  ## mandatory
         maxEvents   = cms.int32(-1),                              ## optional
         outputEvery = cms.uint32(10),                             ## optional
   )

   process.fwliteOutput = cms.PSet(
         fileName = cms.untracked.string('outputHistos.root')      ## mandatory
   )
   
   process.muonAnalyzer = cms.PSet(
     muons = cms.InputTag('cleanPatMuons') ## input for the simple example above
   )


   where the parameters maxEvents and 
   reportAfter are optional. If omitted all events in the file(s) will be looped and no progress
   report will be given. More input files can be given as a vector of strings. Potential histograms 
   per default will be written directely into the file without any furhter directory structure. If
   the class is instantiated with an additional directory string a new directory with the 
   corresponding name will be created and the histograms will be added to this directory.
   With these wrapper classes we have the use case in mind that you keep classes, which easily can 
   be used both within the full framework and within FWLite. 

   NOTE: in the current implementation this wrapper class only supports basic event looping. For 
   anytasks of more complexity we recommend you to start from a FWLiteAnalyzer class from the very 
   beginning and just to stay within FWLite.
*/

namespace fwlite {

  template <class T>
  class AnalyzerWrapper {
  public:
    /// default constructor
    AnalyzerWrapper(const edm::ParameterSet& cfg, std::string analyzerName, std::string directory = "");
    /// default destructor
    virtual ~AnalyzerWrapper(){};
    /// everything which has to be done before the event loop
    virtual void beginJob() { analyzer_->beginJob(); }
    /// everything which has to be done during the event loop. NOTE: the event will be looped inside this function
    virtual void analyze();
    /// everything which has to be done after the event loop
    virtual void endJob() { analyzer_->endJob(); }

  protected:
    /// helper class  for input parameter handling
    fwlite::InputSource inputHandler_;
    /// helper class for output file handling
    fwlite::OutputFiles outputHandler_;
    /// maximal number of events to be processed (-1 means to loop over all event)
    int maxEvents_;
    /// number of events after which the progress will be reported (0 means no report)
    unsigned int reportAfter_;
    /// TFileService for histogram management
    fwlite::TFileService fileService_;
    /// derived class of type BasicAnalyzer
    std::shared_ptr<T> analyzer_;
  };

  /// default contructor
  template <class T>
  AnalyzerWrapper<T>::AnalyzerWrapper(const edm::ParameterSet& cfg, std::string analyzerName, std::string directory)
      : inputHandler_(cfg),
        outputHandler_(cfg),
        maxEvents_(inputHandler_.maxEvents()),
        reportAfter_(inputHandler_.reportAfter()),
        fileService_(outputHandler_.file()) {
    // analysis specific parameters
    const edm::ParameterSet& ana = cfg.getParameter<edm::ParameterSet>(analyzerName.c_str());
    if (directory.empty()) {
      // create analysis class of type BasicAnalyzer
      analyzer_ = std::shared_ptr<T>(new T(ana, fileService_));
    } else {
      // create a directory in the file if directory string is non empty
      TFileDirectory dir = fileService_.mkdir(directory);
      analyzer_ = std::shared_ptr<T>(new T(ana, dir));
    }
  }

  /// everything which has to be done during the event loop. NOTE: the event will be looped inside this function
  template <class T>
  void AnalyzerWrapper<T>::analyze() {
    int ievt = 0;
    std::vector<std::string> const& inputFiles = inputHandler_.files();
    // loop the vector of input files
    fwlite::ChainEvent event(inputFiles);
    for (event.toBegin(); !event.atEnd(); ++event, ++ievt) {
      // break loop if maximal number of events is reached
      if (maxEvents_ > 0 ? ievt + 1 > maxEvents_ : false)
        break;
      // simple event counter
      if (reportAfter_ != 0 ? (ievt > 0 && ievt % reportAfter_ == 0) : false)
        std::cout << "  processing event: " << ievt << std::endl;
      // analyze event
      analyzer_->analyze(event);
    }
  }
}  // namespace fwlite

#endif
