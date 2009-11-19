// system include files
#include <memory>
#include <string>
#include <vector>

// FWCore
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//DQM services
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"
//needed for MessageLogger
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//DataFormat
#include "DataFormats/VertexReco/interface/Vertex.h"

/**
   \class VertexChecker VertexChecker.cc DQM/Physics/plugins/VertexChecker.h

   \brief   Add a one sentence description here...

   Module dedicated to Vertex
   It's an EDAnalyzer
   It uses DQMStore to store the histograms
   in a directory: "Vertex"
*/

class VertexChecker : public edm::EDAnalyzer {
public:
  explicit VertexChecker(const edm::ParameterSet&);
  ~VertexChecker();
    
private:
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  edm::InputTag vertex_;
  DQMStore* dqmStore_;
  std::string outputFileName_;
  bool saveDQMMEs_ ;
  
  //Histograms are booked in the beginJob() method
  std::map<std::string,MonitorElement*> histocontainer_;       // simple map to contain all TH1D.
};
