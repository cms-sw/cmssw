#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "Geometry/Records/interface/PTrackerParametersRcd.h"
#include "CondFormats/GeometryObjects/interface/PTrackerParameters.h"

class TrackerParametersAnalyzer : public edm::EDAnalyzer
{
public:
  explicit TrackerParametersAnalyzer( const edm::ParameterSet& ) {}
  ~TrackerParametersAnalyzer() {}

  virtual void analyze( const edm::Event&, const edm::EventSetup& );
};

void
TrackerParametersAnalyzer::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
   edm::LogInfo("TrackerParametersAnalyzer") << "Here I am";

   edm::ESHandle<PTrackerParameters> ptp;
   iSetup.get<PTrackerParametersRcd>().get( ptp );
   
   for( std::vector<PTrackerParameters::Item>::const_iterator it = ptp->vitems.begin(); it != ptp->vitems.end(); ++it )
     {
       std::cout << it->id << " has " << it->vpars.size() << ": " << std::endl;
       for(  std::vector<int>::const_iterator in = it->vpars.begin(); in !=  it->vpars.end(); ++in )
	 std::cout << *in << "; ";
       std::cout << std::endl;
     }
}

DEFINE_FWK_MODULE(TrackerParametersAnalyzer);
