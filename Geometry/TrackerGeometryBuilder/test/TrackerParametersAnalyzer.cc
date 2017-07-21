#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "Geometry/Records/interface/PTrackerParametersRcd.h"
#include "CondFormats/GeometryObjects/interface/PTrackerParameters.h"
#include "Geometry/CommonDetUnit/interface/GeomDetEnumerators.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerNumberingBuilder/interface/CmsTrackerStringToEnum.h"

class TrackerParametersAnalyzer : public edm::one::EDAnalyzer<>
{
public:
  explicit TrackerParametersAnalyzer( const edm::ParameterSet& ) {}
  ~TrackerParametersAnalyzer() override {}

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}
};

void
TrackerParametersAnalyzer::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
   edm::LogInfo("TrackerParametersAnalyzer") << "Here I am";

   edm::ESHandle<PTrackerParameters> ptp;
   iSetup.get<PTrackerParametersRcd>().get( ptp );

   edm::ESHandle<TrackerGeometry> pDD;
   iSetup.get<TrackerDigiGeometryRecord> ().get (pDD);

   GeometricDet const *gd = pDD->trackerDet();
   GeometricDet::ConstGeometricDetContainer subdetgd = gd->components();
    
   for( GeometricDet::ConstGeometricDetContainer::const_iterator git = subdetgd.begin(); git != subdetgd.end(); ++git )
   {
     std::cout << (*git)->name() << ": " << (*git)->type() << std::endl;
   }
   
   for(const auto & vitem : ptp->vitems)
   {
     std::cout << vitem.id << " is " << pDD->geomDetSubDetector(vitem.id) << " has " << vitem.vpars.size() << ": " << std::endl;
     for(  std::vector<int>::const_iterator in = vitem.vpars.begin(); in !=  vitem.vpars.end(); ++in )
       std::cout << *in << "; ";
     std::cout << std::endl;
   }
   for(int vpar : ptp->vpars)
   {
     std::cout << vpar << "; ";
   }
}

DEFINE_FWK_MODULE(TrackerParametersAnalyzer);
