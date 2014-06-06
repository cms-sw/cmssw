#ifndef GeneratorInterface_RivetInterface_RivetHarvesting
#define GeneratorInterface_RivetInterface_RivetHarvesting

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "Rivet/AnalysisHandler.hh"
#include "Rivet/Tools/RivetYODA.hh"

#include <map>

namespace edm{
  class ParameterSet;
  class Event;
  class EventSetup;
  class InputTag;
}

class RivetHarvesting : public edm::EDAnalyzer
{
  public:
  RivetHarvesting(const edm::ParameterSet&);

  virtual ~RivetHarvesting();

  virtual void beginJob();

  virtual void endJob();  

  virtual void analyze(const edm::Event&, const edm::EventSetup&);

  virtual void beginRun(const edm::Run&, const edm::EventSetup&);

  virtual void endRun(const edm::Run&, const edm::EventSetup&);
  
  private:

  //std::vector<Rivet::DPSXYPoint>  getDPSXYValsErrs(std::string filename, std::string path, std::string name);
  std::vector<YODA::Point2D> getPoint2DValsErrs(std::string filename, std::string path, std::string name);

  Rivet::AnalysisHandler   _analysisHandler;
  std::vector<std::string> _fileNames;
  std::vector<double>      _sumOfWeights;
  std::vector<double>      _crossSections;
  std::vector<double>      _lumis;
  std::string              _outFileName;
  bool                     _isFirstEvent;
  edm::InputTag            _hepmcCollection;
  std::vector<std::string> _analysisNames;
};

#endif
