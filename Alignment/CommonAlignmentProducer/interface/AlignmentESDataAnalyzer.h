
// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "FWCore/Utilities/interface/Exception.h"

// ----------------------------------------------------------------------------
// class decleration

class AlignmentESDataAnalyzer : public edm::EDAnalyzer {
   public:
      explicit AlignmentESDataAnalyzer(const edm::ParameterSet&){};
      ~AlignmentESDataAnalyzer(){};
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
   private:
};

// ----------------------------------------------------------------------------
// method called to produce the data 

void AlignmentESDataAnalyzer::analyze(const edm::Event& iEvent, 
  const edm::EventSetup& iSetup)
{
  using namespace edm;

  ESHandle<TrackerGeometry> pData;
  iSetup.get<TrackerDigiGeometryRecord>().get(pData);
}

