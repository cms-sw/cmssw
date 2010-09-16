#ifndef PhysicsTools_UtilAlgos_interface_FWLiteAnalyzerWrapper_h
#define PhysicsTools_UtilAlgos_interface_FWLiteAnalyzerWrapper_h

/**
  \class    FWLiteAnalyzerWrapper FWLiteAnalyzerWrapper.h "PhysicsTools/UtilAlgos/interface/FWLiteAnalyzerWrapper.h"
  \brief    Implements a wrapper around an FWLite-friendly analyzer to "convert" it into a full EDAnalyzer


  \author Salvatore Rappoccio
  \version  $Id: FWLiteAnalyzerWrapper.h,v 1.1 2010/05/02 04:53:06 srappocc Exp $
*/


#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include <boost/shared_ptr.hpp>

namespace edm {

template<class T>
class FWLiteAnalyzerWrapper : public EDAnalyzer {

 public:

  /// Pass the parameters to analyzer_
 FWLiteAnalyzerWrapper(const edm::ParameterSet& pset)
  {
    edm::Service<TFileService> fileService;

    analyzer_  = boost::shared_ptr<T>( new T( pset, *fileService) );

  }

  /// Destructor does nothing
  virtual ~FWLiteAnalyzerWrapper() {}
 

  /// Pass the event to the analyzer. NOTE! We can't use the eventSetup in FWLite so ignore it.
  virtual void analyze( edm::Event const & event, const edm::EventSetup& eventSetup)
  {
    analyzer_->analyze(event); 
  }

  /// Pass the begin and endJobs to the analyzer. 
  virtual void beginJob() {  analyzer_->beginJob(); }
  virtual void endJob() {  analyzer_->endJob(); }

 protected:
  boost::shared_ptr<T> analyzer_;
};

}

#endif
