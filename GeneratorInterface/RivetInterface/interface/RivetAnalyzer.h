#ifndef GeneratorInterface_RivetInterface_RivetAnalyzer
#define GeneratorInterface_RivetInterface_RivetAnalyzer

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "Rivet/AnalysisHandler.hh"

namespace edm{
  class ParameterSet;
  class Event;
  class EventSetup;
  class InputTag;
}

class RivetAnalyzer : public edm::EDAnalyzer
{
  public:
  RivetAnalyzer(const edm::ParameterSet&);

  virtual ~RivetAnalyzer();

  virtual void beginJob();

  virtual void endJob();  

  virtual void analyze(const edm::Event&, const edm::EventSetup&);

  virtual void beginRun(const edm::Run&, const edm::EventSetup&);

  virtual void endRun(const edm::Run&, const edm::EventSetup&);
  
  private:

  edm::InputTag            _hepmcCollection;
  Rivet::AnalysisHandler   _analysisHandler;   
  bool                     _isFirstEvent;
  std::string              _outFileName; 
};

#endif
